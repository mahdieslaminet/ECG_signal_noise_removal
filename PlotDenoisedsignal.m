function PlotDenoisedsignal(varargin)
    bestSNRidx = varargin{1};
    worstSNRidx = varargin{2};
    plotRange = 2000:3000;
    labels = ["Noisy","Denoised (advDenoiser)","Clean","Denoised (wavDenoiser)"];
    
    figure
    hold on
    for i = 3:nargin
        signal = varargin{i};
        plot(plotRange,(signal(bestSNRidx,plotRange)));
    end

   
    legend(labels(1:nargin-2),Location="southoutside",Orientation = "horizontal",NumColumns=2)
    title("Denoised Signal with Best SNR")
    hold off

    figure
    hold on
    for i = 3:nargin
        signal = varargin{i};
        plot(plotRange,(signal(worstSNRidx,plotRange)));
    end
    legend(labels(1:nargin-2),Location="southoutside",Orientation = "horizontal",NumColumns=2)
    title("Denoised Signal with Worst SNR")
    hold off

end