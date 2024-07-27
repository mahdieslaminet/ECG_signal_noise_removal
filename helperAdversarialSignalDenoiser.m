classdef helperAdversarialSignalDenoiser < handle
% Adversarial Learning Signal Denoiser
%   This class is only for use in the "Denoise Signals with an
%   Adversarial Learning Denoiser Model" example. It may change or be
%   removed in a future release.
%   
%   denoiser = helperAdvasarialSignalDenoiser() creates a adversarial
%   denoiser.
%
%   YDs = denoise(denoiser,XDs) denoises noisy signals in datastore XDs and
%   returns denoised signals in datastore YDs.
%   
%   train(denoiser,trainDs,'PARAM1',VAL1,'PARAM2',VAL2,...) trains
%   the denoiser on dataset in datastore trainDs. The optional parameters
%   can be specified here using name-value arguments:
%
%       'Normalization'       - If it is true, zerocenter normalization is 
%                               applied before data is forward propagated 
%                               through the input layer.  
%
%       'MaxEpochs'           - The maximum number of epochs that will be
%                               used for training. The default is 30.
%
%       'MiniBatchSize'       - The size of the mini-batch used for each
%                               training iteration. The default is 128.
%
%       'Plots'               - If it is true, the training progress plot
%                               will be displayed.
%
%       'Verbose'             - If this is set to true, information on
%                               training progress will be printed to the
%                               command window.
%
%       'VerboseFrequency'    - This parameter only applies when 'Verbose' 
%                               is set to true. It specifies the number of
%                               iterations between printing to the command
%                               window. The default is 10.
%
%       'ValidationData'      - Data used to do validation during training.
%
%       'ValidationFrequency' - Number of iterations between evaluations of
%                               validation metrics. This parameter only 
%                               applies when if you also specify 
%                               'ValidationData'. The default is 5.
%
%       'ExecutionEnvironment'- The execution environment for the
%                               network. This determines what hardware
%                               resources are used to train the
%                               network. To use a GPU, you must have 
%                               Parallel Computing Toolbox(TM) and a 
%                               supported GPU device. The default is 'auto'.
%                                 - 'auto' - Use a GPU if it is
%                                   available, otherwise use the CPU.
%                                 - 'gpu' - Use the GPU.
%                                 - 'cpu' - Use the CPU.
%
%   Copyright 2021 The Mathworks, Inc.
    
    properties
        % Model States
        IsTrained
        
        % TrainingOptions
        ExecutionEnvironment
        MaxEpochs
        MiniBatchSize
        Plots
        Verbose
        VerboseFrequency
        ValidationFrequency
        SignalLength
    end

    properties (SetAccess = private)
        DoTraining = false;
        DoValidation = false;

        % Model Paramters
        Normalization
        ParametersEnDecoder
        ParametersDiscriminator

        % Solver States
        AccumUpdateEnDecoder
        AccumGradEnDecoder
        AccumUpdateDiscriminator
        AccumGradDiscriminator
    end


    methods
        %------------------------------------------------------------------
        function this = helperAdversarialSignalDenoiser(signalLegnth)
            arguments
                signalLegnth {mustBePositive,mustBeInteger}
            end
            this.IsTrained = false;
            this.SignalLength = signalLegnth;
            
            % After signal length set, initialize the parameters
            this.initializeParametersEnDecoder()
            this.initializeParametersDiscriminator()
        end

        %------------------------------------------------------------------
        function train(this,trainDs,NVargs)
            arguments
                this
                trainDs
                NVargs.ValidationData = []
                NVargs.ValidationFrequency {mustBePositive,mustBeInteger} = 5;
                NVargs.Normalization {mustBeNumericOrLogical} = false;
                NVargs.MiniBatchSize {mustBePositive,mustBeInteger} = 32
                NVargs.MaxEpochs {mustBePositive,mustBeInteger} = 10       
                NVargs.Plots {mustBeNumericOrLogical} = true;
                NVargs.Verbose {mustBeNumericOrLogical} = true;
                NVargs.VerboseFrequency {mustBePositive,mustBeInteger} = 10
                NVargs.ExecutionEnvironment {mustBeText,mustBeMember(NVargs.ExecutionEnvironment,{'cpu','gpu','auto'})} = "auto"
            end
            this.Normalization = NVargs.Normalization;
            this.ValidationFrequency = NVargs.ValidationFrequency;
            this.MiniBatchSize = NVargs.MiniBatchSize;
            this.MaxEpochs = NVargs.MaxEpochs;
            this.Plots = NVargs.Plots;
            this.Verbose = NVargs.Verbose;
            this.VerboseFrequency = NVargs.VerboseFrequency;
            this.ExecutionEnvironment = NVargs.ExecutionEnvironment;
           
            mbqTrain = minibatchqueue(trainDs,...
                "MiniBatchSize",this.MiniBatchSize,...
                "MiniBatchFcn", @(X,Y)this.processBatch(X,Y),...
                "MiniBatchFormat",{'SCB','SCB'},...
                "OutputAsDlarray",[1 1],...
                "OutputEnvironment",this.ExecutionEnvironment,...
                "DispatchInBackground",0);

            if ~isempty(NVargs.ValidationData)
                this.DoValidation = true;
                ValidationData = readall(NVargs.ValidationData); 
                ValidationData = cat(1,ValidationData{:});
            end

            if this.Plots
                figure
                lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
                lineLossValidation = animatedline( ...
                    'LineStyle','--', ...
                    'Marker','o', ...
                    'MarkerFaceColor','black');
                xlabel("Iteration")
                ylabel("Loss in Log Scale")
                legend(["Training Loss","Validation Loss"])
                grid on
            end
            
            % Loop over epochs.
            start = tic;
            iteration = 0;
            if this.DoValidation
                    [XValidation, TValidation] = this.processBatch(ValidationData(:,1),ValidationData(:,2));
                    dlXValidation = dlarray(XValidation,'SCB');
                    TValidation = dlarray(TValidation,'SCB');
            end

            % Set the object as trained
            this.IsTrained = true;
            % Loop over the epochs
            for epoch = 1:this.MaxEpochs
                shuffle(mbqTrain);
                
                if this.Plots && this.DoValidation && (epoch == 1 || mod(epoch,this.ValidationFrequency) == 0)                 
                    dlY = this.modelEnDecoder(dlXValidation, this.ParametersEnDecoder);
                    lossValidation = mse(dlY,TValidation)/size(dlY,1);
                    lossValidation = double(extractdata(lossValidation));
                    addpoints(lineLossValidation,iteration,log(lossValidation));
                    drawnow
                end

                % Loop over mini-batches.
                while hasdata(mbqTrain)                 
                    [dlX,dlT] = next(mbqTrain);
                    iteration = iteration + 1;
                    
                    % Compute loss and gradients.
                    [gradientsEnDecoder,gradientsDiscriminator,loss] = dlfeval(@this.modelGradientsAndLoss,dlX,dlT,this.ParametersEnDecoder,this.ParametersDiscriminator);
        
                    updateFcn = @this.adadelta;
                    % Update the encoder/decoder network parameters.
                    [this.ParametersEnDecoder,this.AccumGradEnDecoder,this.AccumUpdateEnDecoder] = dlupdate(updateFcn,this.ParametersEnDecoder,gradientsEnDecoder,this.AccumGradEnDecoder,this.AccumUpdateEnDecoder);
        
                    % Update the discriminator network parameters.
                    [this.ParametersDiscriminator,this.AccumGradDiscriminator,this.AccumUpdateDiscriminator] = dlupdate(updateFcn,this.ParametersDiscriminator,gradientsDiscriminator,this.AccumGradDiscriminator,this.AccumUpdateDiscriminator);
                    if this.Plots
                        D = duration(0,0,toc(start),'Format','hh:mm:ss');
                        addpoints(lineLossTrain,iteration,log(loss))
                        title("Epoch: " + epoch + ", Elapsed: " + string(D))
                    end
                end

                if this.Verbose && (epoch == 1 || mod(epoch,this.VerboseFrequency) == 0) 
                    disp("Training loss after epoch " + epoch + ": " + sprintf('%0.5g',loss));
                end
                title("Epoch: " + epoch + ", Elapsed Time: " + string(D))
                drawnow
            end

            % Gather parameters back for compatability
            this.ParametersEnDecoder = dlupdate(@gather, this.ParametersEnDecoder);
            this.AccumGradEnDecoder = dlupdate(@gather, this.AccumGradEnDecoder);
            this.AccumUpdateEnDecoder = dlupdate(@gather, this.AccumUpdateEnDecoder);

            this.ParametersDiscriminator = dlupdate(@gather,this.ParametersDiscriminator);
            this.AccumGradDiscriminator = dlupdate(@gather,this.AccumGradDiscriminator);
            this.AccumUpdateDiscriminator = dlupdate(@gather,this.AccumUpdateDiscriminator);
        end

        %------------------------------------------------------------------
        function testYDS = denoise(this,testDs,NVargs)
            arguments
                this
                testDs
                NVargs.MiniBatchSize {mustBePositive,mustBeInteger} = 32
                NVargs.ExecutionEnvironment {mustBeText,mustBeMember(NVargs.ExecutionEnvironment,{'cpu','gpu','auto'})} = "auto"
            end
            checkIsTrained(this);

            % The 'S' here represents the time dimension of the signal
            mbqTest = minibatchqueue(testDs,1,...
                "MiniBatchSize",NVargs.MiniBatchSize,...
                "MiniBatchFcn", @(X)this.preprocessData(X,this.Normalization),...
                "MiniBatchFormat",{'SCB'},...
                "OutputAsDlarray",1,...
                "OutputEnvironment",NVargs.ExecutionEnvironment,...
                "DispatchInBackground",0);

            testYcell = {};
            while hasdata(mbqTest)
                dltestX = next(mbqTest);
                dltestY = this.modelEnDecoder(dltestX, this.ParametersEnDecoder);
                testY = squeeze(gather(extractdata(dltestY)))';
                testYcell = [testYcell;mat2cell(testY,ones([size(testY,1),1]))];
            end

            testYDS = signalDatastore(testYcell);
        end

        %------------------------------------------------------------------
        function [XOut,YOut] = processBatch(this,XIn,YIn)
            XOut = this.preprocessData(XIn, this.Normalization);
            YOut = this.preprocessData(YIn, false);
        end

        %------------------------------------------------------------------
        function reset(this)
            this.IsTrained = false;
            this.initializeParametersEnDecoder();
            this.initializeParametersDiscriminator();
        end

        %------------------------------------------------------------------
        function initializeParametersEnDecoder(this)
            initializeGlorot = @this.initializeGlorot;
            initializeZeros = @this.initializeZeros;

            % Intialize the learnable parameters for encoder and decoder part
            this.ParametersEnDecoder.conv1.Weights = initializeGlorot([3 1 128],128,1);
            this.ParametersEnDecoder.conv1.Bias = initializeZeros([1 128]);
            this.ParametersEnDecoder.conv2.Weights = initializeGlorot([3 128 128],128,128);
            this.ParametersEnDecoder.conv2.Bias = initializeZeros([1 128]);
            this.ParametersEnDecoder.conv3.Weights = initializeGlorot([3 128 128],128,128);
            this.ParametersEnDecoder.conv3.Bias = initializeZeros([1 128]);
            this.ParametersEnDecoder.conv4.Weights = initializeGlorot([3 128 128],128,128);
            this.ParametersEnDecoder.conv4.Bias = initializeZeros([1 128]);
            
            this.ParametersEnDecoder.deconv1.Weights = initializeGlorot([3 128 128],128,128);
            this.ParametersEnDecoder.deconv1.Bias = initializeZeros([1 128]);
            this.ParametersEnDecoder.conv5.Weights = initializeGlorot([3 128 128],128,128);
            this.ParametersEnDecoder.conv5.Bias = initializeZeros([1 128]);
            this.ParametersEnDecoder.deconv2.Weights = initializeGlorot([3 128 128],128,128);
            this.ParametersEnDecoder.deconv2.Bias = initializeZeros([1 128]);
            this.ParametersEnDecoder.conv6.Weights = initializeGlorot([3 128 128],128,128);
            this.ParametersEnDecoder.conv6.Bias = initializeZeros([1 128]);
            this.ParametersEnDecoder.deconv3.Weights = initializeGlorot([3 128 128],128,128);
            this.ParametersEnDecoder.deconv3.Bias = initializeZeros([1 128]);
            this.ParametersEnDecoder.conv7.Weights = initializeGlorot([3 128 128],128,128);
            this.ParametersEnDecoder.conv7.Bias = initializeZeros([1 128]);
            this.ParametersEnDecoder.conv8.Weights = initializeGlorot([3 128 1],1,128);
            this.ParametersEnDecoder.conv8.Bias = initializeZeros([1 1]);

            this.AccumGradEnDecoder = this.ParametersEnDecoder;
            this.AccumUpdateEnDecoder = this.ParametersEnDecoder;
            f = fields(this.ParametersEnDecoder);
            for ii=1:numel(f)
                this.AccumGradEnDecoder.(f{ii}).Weights(:) = 0;
                this.AccumUpdateEnDecoder.(f{ii}).Weights(:) = 0;
            end
        end

        %------------------------------------------------------------------
        function initializeParametersDiscriminator(this)
            % Intialize the learnable parameters for encoder and decoder part
            initializeGlorot = @this.initializeGlorot;
            initializeZeros = @this.initializeZeros;
        
            this.ParametersDiscriminator.conv9.Weights = initializeGlorot([3 128 1],1,128);
            this.ParametersDiscriminator.conv9.Bias = initializeZeros([1 1]);
            this.ParametersDiscriminator.fc1.Weights = initializeGlorot([this.SignalLength 150],150,this.SignalLength);
            this.ParametersDiscriminator.fc1.Bias = initializeZeros([1 150]);
            this.ParametersDiscriminator.fc2.Weights = initializeGlorot([150 150],150,150);
            this.ParametersDiscriminator.fc2.Bias = initializeZeros([1 150]);
            this.ParametersDiscriminator.fc3.Weights = initializeGlorot([150 2],2,150);
            this.ParametersDiscriminator.fc3.Bias = initializeZeros([1 2]);
            
            this.AccumGradDiscriminator = this.ParametersDiscriminator;
            this.AccumUpdateDiscriminator = this.ParametersDiscriminator;
            f = fields(this.ParametersDiscriminator);
            for ii=1:numel(f)
                this.AccumGradDiscriminator.(f{ii}).Weights(:) = 0;
                this.AccumUpdateDiscriminator.(f{ii}).Weights(:) = 0;
            end
        end


        %------------------------------------------------------------------
        function [dlY1,Z4] = modelEnDecoder(~,dlX,parameters)
            % encoder
            Z1 = dlconv(dlX,parameters.conv1.Weights,parameters.conv1.Bias,'Padding','same');
            Z1 = relu(Z1);
            Z2 = dlconv(Z1,parameters.conv2.Weights,parameters.conv2.Bias,'Padding','same','DilationFactor',3);
            Z2 = relu(Z2);
            Z3 = dlconv(Z2,parameters.conv3.Weights,parameters.conv3.Bias,'Padding','same','DilationFactor',3);
            Z3 = relu(Z3);
            Z4 = dlconv(Z3,parameters.conv4.Weights,parameters.conv4.Bias,'Padding','same','DilationFactor',6);
            Z4 = relu(Z4);
            % decoder
            Z5 = dltranspconv(Z4,parameters.deconv1.Weights,parameters.deconv1.Bias,'DilationFactor',6,'Cropping','same');
            Z5 = relu(Z5);
            resBlock1 = Z3 + Z5;
            Z6 = dlconv(resBlock1,parameters.conv5.Weights,parameters.conv5.Bias,'Padding','same');
            Z6 = relu(Z6);
            Z7 = dltranspconv(Z6,parameters.deconv2.Weights,parameters.deconv2.Bias,'DilationFactor',3,'Cropping','same');
            Z7 = relu(Z7);
            resBlock2 = Z2 + Z7;
            Z8 = dlconv(resBlock2,parameters.conv6.Weights,parameters.conv6.Bias,'Padding','same');
            Z8 = relu(Z8);
            Z9 = dltranspconv(Z8,parameters.deconv3.Weights,parameters.deconv3.Bias,'DilationFactor',3,'Cropping','same');
            Z9 = relu(Z9);
            resBlock3 = Z1 + Z9;
            Z10 = dlconv(resBlock3,parameters.conv7.Weights,parameters.conv7.Bias,'Padding','same');
            Z10 = relu(Z10);
            dlY1 = dlconv(Z10,parameters.conv8.Weights,parameters.conv8.Bias,'Padding','same');
        end

        %------------------------------------------------------------------
        function dlY2 = modelDiscriminator(~,Z4,parameters)
            % discriminator
            Z11 = dlconv(Z4,parameters.conv9.Weights,parameters.conv9.Bias,'Padding','same');
            Z12 = fullyconnect(Z11,parameters.fc1.Weights,parameters.fc1.Bias);
            Z12 = relu(Z12);
            Z13 = fullyconnect(Z12,parameters.fc2.Weights,parameters.fc2.Bias);
            Z13 = relu(Z13);
            Z14 = fullyconnect(Z13,parameters.fc3.Weights,parameters.fc3.Bias);
            dlY2 = sigmoid(Z14);
        end

        %------------------------------------------------------------------
        function [gradientsEnDecoder, gradientsDiscriminator, lossMSE, lossEncoderDeCoder, lossDiscriminator] = modelGradientsAndLoss(this,dlX,dlT,parametersEnDecoder,parametersDiscriminator)
            % Encode noise signal
            [dlYR1, Z1] = modelEnDecoder(this,dlX,parametersEnDecoder);
            % Encode clean signal
            [~, Z2] = modelEnDecoder(this,dlT,parametersEnDecoder);
        
            % classify encoded noise signal
            dlYC1 = modelDiscriminator(this,Z1,parametersDiscriminator);
            % classify encoded clean signal
            dlYC2 = modelDiscriminator(this,Z2,parametersDiscriminator);
            
            % 1st row clean
            % 2nd row noisy
            % Normalized regression mes loss for encoder&decoder
            loss1 = mse(dlYR1,dlT)/this.SignalLength; 
            
            % Classfication loss for encoder 
            % The more deceives the discriminator and be calssified as 
            % 'encoded from clean', the lower loss.
            loss4 = -mean(log(dlYC1(1,:)+eps)); 
            lossEncoderDeCoder = loss1 + loss4;
            
            % Classfication loss for discriminator
            loss2 = -mean(log(dlYC1(2,:)+eps));
            loss3 =- mean(log(dlYC2(1,:)+eps));
            lossDiscriminator = loss2 + loss3;
        
            gradientsEnDecoder = dlgradient(lossEncoderDeCoder,parametersEnDecoder);
            gradientsDiscriminator = dlgradient(lossDiscriminator,parametersDiscriminator);
            
            lossMSE = double(gather(extractdata(loss1)));
            lossEncoderDeCoder = double(gather(extractdata(lossEncoderDeCoder)));
            lossDiscriminator = double(gather(extractdata(lossDiscriminator)));
        end

        function saveParameters(this,filename)
            arguments
                this
                filename {mustBeTextScalar} = 'adversarialLearningDenoiserModelParameters'
            end
            checkIsTrained(this);
            % Model Paramters
            SignalLength = this.SignalLength;
            Normalization = this.Normalization;
            ParametersEnDecoder = this.ParametersEnDecoder;
            ParametersDiscriminator = this.ParametersDiscriminator;

            % Solver States
            AccumUpdateEnDecoder = this.AccumUpdateEnDecoder;
            AccumGradEnDecoder = this.AccumGradEnDecoder;
            AccumUpdateDiscriminator = this.AccumUpdateDiscriminator;
            AccumGradDiscriminator = this.AccumGradDiscriminator;
            save(filename, ...
                'SignalLength', ...
                'Normalization',...
                'ParametersDiscriminator', ...
                'ParametersEnDecoder', ...
                'AccumUpdateEnDecoder', ...
                'AccumGradEnDecoder', ...
                'AccumUpdateDiscriminator', ...
                'AccumGradDiscriminator');
        end

        function loadParameters(this,parametersFile)
            load(parametersFile);
            % Model Paramters
            this.SignalLength = SignalLength;
            this.Normalization = Normalization;
            this.ParametersEnDecoder = ParametersEnDecoder;
            this.ParametersDiscriminator = ParametersDiscriminator;
            
            % Solver States
            this.AccumUpdateEnDecoder = AccumUpdateEnDecoder;
            this.AccumGradEnDecoder = AccumGradEnDecoder;
            this.AccumUpdateDiscriminator = AccumUpdateDiscriminator;
            this.AccumGradDiscriminator = AccumGradDiscriminator;
            
            % Set the object as trained
            this.IsTrained = true;
        end


    end
    methods (Access = protected)
        %------------------------------------------------------------------
        function checkIsTrained(this)
            if ~this.IsTrained
                error('Denoiser is not trained. Use trainNet method to train.')
            end
        end

    end


    methods(Static)
        %------------------------------------------------------------------
        function weights = initializeGlorot(sz,numOut,numIn,className)
            arguments
                sz
                numOut
                numIn
                className = 'single'
            end
            Z = 2*rand(sz,className) - 1;
            bound = sqrt(6 / (numIn + numOut));

            weights = bound * Z;
            weights = dlarray(weights,'SCU');
        end

        %------------------------------------------------------------------
        function parameter = initializeZeros(sz,className)
            arguments
                sz
                className = 'single'
            end
            parameter = zeros(sz,className);
            parameter = dlarray(parameter);

        end
        %------------------------------------------------------------------
        function dataOut = preprocessData(dataIn,doNormalization)
            % If X and Y are not numerical but a cell array
            if ~isnumeric(dataIn)
                dataOut = cat(1,dataIn{:});
            else
                dataOut = dataIn;
            end
            if doNormalization == true
%                 dataOut = normalize(dataOut,2);
                dataOut = dataOut-mean(dataOut,2);
            end
            dataOut = permute(dataOut,[2 3 1]);
        end

        %------------------------------------------------------------------
        function [parameter,accumGrad,accumUpdate] = adadelta(p,g,accumGrad,accumUpdate,rho,epsilon)
            arguments
                p
                g
                accumGrad
                accumUpdate
                rho(1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(rho, 0), mustBeLessThan(rho, 1)} = 0.9;
                epsilon(1,1) {mustBeNumeric, mustBeFinite, mustBePositive} = 1e-8;
            end
            accumGrad = rho * accumGrad + (1 - rho) * g.^2;
            update = -sqrt((accumUpdate + epsilon) ./ (accumGrad + epsilon)).*g;
            accumUpdate = rho * accumUpdate + (1 - rho) * update.^2;
            parameter = p + update;
        end

    end
end

