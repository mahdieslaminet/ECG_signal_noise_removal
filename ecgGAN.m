clc
clear
close

datasetFolder = fullfile(tempdir,"ecg-id-database-1.0.0");
if ~isfolder(datasetFolder)
    loc = websave(tempdir,"https://physionet.org/static/published-projects/ecgiddb/ecg-id-database-1.0.0.zip");
    unzip(loc,tempdir);
end

sds = signalDatastore(datasetFolder, ...
                      IncludeSubfolders = true, ...
                      FileExtensions = ".dat", ...
                      ReadFcn = @ReadSignalData);
rng("default")

subjectIds = unique(extract(sds.Files,"Person_"+digitsPattern));
testSubjects = contains(sds.Files,subjectIds(randperm(numel(subjectIds),10)));
testDs = subset(sds,testSubjects);

sds = signalDatastore(datasetFolder, ...
                      IncludeSubfolders = true, ...
                      FileExtensions = ".dat", ...
                      ReadFcn = @ReadSignalData);
rng("default")

subjectIds = unique(extract(sds.Files,"Person_"+digitsPattern));
testSubjects = contains(sds.Files,subjectIds(randperm(numel(subjectIds),10)));
testDs = subset(sds,testSubjects);

trainAndValDs = subset(sds,~testSubjects);
trainAndValDs = shuffle(trainAndValDs);
[trainInd,valInd] = dividerand(1:numel(trainAndValDs.Files),0.8,0.2,0);
trainDs = subset(trainAndValDs,trainInd);
validDs = subset(trainAndValDs,valInd);

sampleSignal = preview(trainDs);
signalLength = length(sampleSignal{1});
advDenoiser = helperAdversarialSignalDenoiser(signalLength);

doTrain = true;
if doTrain
    train(advDenoiser,trainDs,...
        ValidationData = validDs, ...
        MaxEpochs = 100, ...
        MiniBatchSize = 32, ...
        Plots = true, ...
        Normalization = true);
end
denoisedSignalsDs = denoise(advDenoiser,testDs, ...
    "MiniBatchSize",32, ...
    "ExecutionEnvironment","auto");
testData = readall(testDs);
denoisedSignals = readall(denoisedSignalsDs);
denoisedSignals = cat(1,denoisedSignals{:});

noisySignals = cellfun(@(x) x(1),testData);
noisySignals = cat(1,noisySignals{:});

cleanSignals = cellfun(@(x) x(2),testData);
cleanSignals = cat(1,cleanSignals{:});

N = size(cleanSignals,1);
snrsNoisy = zeros(N,1);
snrsDenoised = zeros(N,1);
snrsWaveletDenoised = zeros(N,1);
for i = 1:N
    snrsNoisy(i) = snr(cleanSignals(i,:),cleanSignals(i,:)-noisySignals(i,:));
end
for i = 1:N
    snrsDenoised(i) = snr(cleanSignals(i,:),cleanSignals(i,:)-denoisedSignals(i,:));
end

SNRs = [snrsNoisy,snrsDenoised];


bins = -10:2:16;
count = zeros(2,length(bins)-1);
for i =1:2
    count(i,:) = histcounts(SNRs(:,i),bins);
end

bar(bins(1:end-1),count,"stack");
legend(["Noisy","Denoised (advDenoiser)"],"Location","northwest")
title("SNR of the Noisy and Denoised Signals")
xlabel("SNR (dB)")
ylabel("Number of Samples")
grid on

[bestSNR,bestSNRIdx] = max(snrsDenoised)
[worstSNR,worstSNRIdx] = min(snrsDenoised)
PlotDenoisedsignal(bestSNRIdx,worstSNRIdx,noisySignals,denoisedSignals,cleanSignals)

noisySignalsNormalized = noisySignals - mean(noisySignals,2);
waveletDenoisedSignals = wdenoise(double(noisySignalsNormalized),...
    Wavele = "sym8", ...
    ThresholdRule = "soft", ...
    NoiseEstimate = "LevelDependent");
for i = 1:N
    snrsWaveletDenoised(i) = snr(cleanSignals(i,:),cleanSignals(i,:)-waveletDenoisedSignals(i,:));
end
SNRs = [snrsNoisy,snrsDenoised snrsWaveletDenoised];

bins = -10:2:16;
count = zeros(3,length(bins)-1);
for i =1:3
    count(i,:) = histcounts(SNRs(:,i),bins);
end
bar(bins(1:end-1),count,"stack");
legend(["Noisy","Denoised (advDenoiser)","Denoised (wavDenoiser)"],"Location","best")
title("SNR of the Noisy and Denoised Signals")
xlabel("SNR (dB)")
ylabel("Number of Samples")
grid on
PlotDenoisedsignal(bestSNRIdx,worstSNRIdx,noisySignals,denoisedSignals,cleanSignals,waveletDenoisedSignals)
