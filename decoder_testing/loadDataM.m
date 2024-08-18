function [time,isUpdating,predictionWeight, ...
    yDesired,probaPredicted,threshold,yPredicted, raw, X, trigs, events, xLatent] = loadDataM(datafolder, datafiles)

    % Get number of files
    nFiles = length(datafiles);

    % Initialize the fields with approximately the right duration
    % (what's unknown at this point is the duration of the last .dt5 file)

    if nFiles == 0
        % If no files, return immediately
        warning('No data files provided.');
        return;
    else
        disp('Number of files concatenated: ')
        disp(nFiles)
    end
        
    trigs=[];
    for i = 1:nFiles
        
        % Load the file
        S=load(fullfile(datafolder,datafiles{i}),'-mat');
        ExperimentData=S.ExperimentData;

        % Get number of points
        nPoints = length(ExperimentData);

        % Get first timestamp and size of the feature space
        if i == 1
            startTime = ExperimentData{1}.Time;
            nstates = length(ExperimentData{1}.State);
            n_chans = size(ExperimentData{1}.x,3);
            try
            n_trigs = size(ExperimentData{1}.AdditionalChannels,1);
            trigs = zeros(n_trigs,3000*(nFiles-1));
            end
            time = zeros(1,3000*(nFiles-1));
            isUpdating = zeros(1,3000*(nFiles-1)); 
            predictionWeight = zeros(nstates,3000*(nFiles-1));   
            yDesired = zeros(nstates,3000*(nFiles-1));
            events = {};
            yPredicted = zeros(nstates,3000*(nFiles-1));
            probaPredicted = zeros(nstates,3000*(nFiles-1));
            threshold = zeros(1,3000*(nFiles-1));
            raw = zeros(n_chans,3000*(nFiles-1)*59);           
            X = zeros([size(ExperimentData{1}.x) 3000*(nFiles-1)]);
            xLatent = zeros([size(ExperimentData{1}.xLatent,2) 3000*(nFiles-1)]);
        end

        % Adapt size of each field depending on the duration of the last .dt5
        if nPoints < 3000
            time = [time zeros(1,nPoints)];
            isUpdating = [isUpdating zeros(1,nPoints)];
            predictionWeight = [predictionWeight zeros(nstates,nPoints)];
            yDesired = [yDesired zeros(nstates,nPoints)];
            probaPredicted = [probaPredicted zeros(nstates,nPoints)];
            threshold = [threshold zeros(1,nPoints)];
            raw=[raw zeros(n_chans,nPoints*59)];
            try
            trigs = [trigs zeros(n_trigs,nPoints)];
            end
            X=cat(4,X, zeros(size(ExperimentData{1}.x)));
            xLatent = cat(2,xLatent, zeros(size(ExperimentData{1}.xLatent,2)));
        end

        % Get fields
        for j = 1:nPoints
            sample = (i-1)*3000+j;
            time(sample) = seconds(ExperimentData{j}.Time-startTime);
            isUpdating(sample) = ExperimentData{j}.IsUpdating;
            %predictionWeight(:,sample) = ExperimentData{j}.ScenarioSupplementaryData.PredictionWeight;
            yDesired(:,sample) = ExperimentData{j}.State;
            events{sample} = ExperimentData{j}.ScenarioSupplementaryData{1,1}.MotionState;
            probaPredicted(:,sample) = ExperimentData{j}.AlphaPredicted{1};
            raw(:,1+(i-1)*3000*59+59*(j-1):(i-1)*3000*59+59*j) = ExperimentData{j}.RawDataBuffer; 
            X(:,:,:,1+(i-1)*3000+(j-1):(i-1)*3000+j) = ExperimentData{j}.x;
            xLatent(:,1+(i-1)*3000+(j-1):(i-1)*3000+j) = squeeze(ExperimentData{j}.xLatent); 
            try
            trigs(:,1+(i-1)*3000+(j-1):(i-1)*3000+j) = sum((ExperimentData{j}.AdditionalChannels)>1);
            end
            %threshold(sample) = ExperimentData{j}.ScenarioSupplementaryData.SpeedThresholds;
            % I commented this line ^^^, maybe it will screw up
            % things!!!!!! (TODO) .. Martin :)
        end
       
    end
    yPredicted = 0; %probaPredicted > threshold;
end
