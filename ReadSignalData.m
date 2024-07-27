function [dataOut,infoOut] = ReadSignalData(filename)
    fid = fopen(filename,"r");
    % 1st row : raw data, 2nd row : filtered data
    data = fread(fid,[2 Inf],"int16=>single");
    fclose(fid);
    fid = fopen(replace(filename,".dat",".hea"),"r");
    header = textscan(fid,"%s%d%d%d",2,"Delimiter"," ");
    fclose(fid);
    gain = single(header{3}(2));
    dataOut{1} = data(1,:)/gain; % noisy, raw data
    dataOut{2} = data(2,:)/gain; % filtered, clean data
    infoOut.SampleRate = header{3}(1);
end
