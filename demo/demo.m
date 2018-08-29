%% IMPORT TOY DATA
load('demo_data');
% define input data (X) and data labels (Y)
X = data(:, 1:end-1); 
Y = data(:, end);
% plot in 3D to visualise
figure(1)
scatter3(X(:,1),X(:,2),X(:,3),20,Y,'.');
colormap('jet'); colorbar;

%% DEMO 1
% single test

% --------- DEFINE TEST SET
[idx,~,test_idx] = dividerand(length(data),2/3,0,1/3);
% available data to build the training-set
X = data(idx, 1:end-1); 
Y = data(idx, end); 
% test-set
x_test = data(test_idx, 1:end-1);
y_test = data(test_idx, end);



% ACTIVE LEARNING

% --------- CLUSTER the pool of available unlabelled data
[u, ch] = h_cluster(X);


% --------- ACTIVE LEARNING -- build the training set -- the DH learner
n = 180; % label budget
B = 3; % batch size
t = n/3; % number of runs
% the DH learner
[xl, z] = DH_AL(u, ch, B, t, Y);

% define the training-set with the DH results
train_idx = xl(:, 1);
x_train  = X(train_idx, :);
y_train = xl(:, 2);


% --------- CLASSIFICATION -- train/predict with niave bayes classification
y_pred = NB(x_train, y_train, x_test);
% calculate classification accuracy
acc = sum(y_pred == y_test)/length(y_test);
fprintf('\n ACTIVE LEARNING classification accuracy: %.2f %%', acc);


% --------- PLOT the training set built by the active learner
% queried data
z_idx = z(:,1); % output of the DH learner is indexed
Z = X(z_idx,:);
% plot
figure(2)
scatter3(x_train(:,1),x_train(:,2),x_train(:,3),20,y_train,'.');
hold on
scatter3(Z(:,1),Z(:,2),Z(:,3),10,'ko');
colormap('jet'); colorbar;
hold off



% PASSIVE LEARNING comparison -- random sample training

% define the training-set by a random sample
train_idx = randperm(size(X,1), n);
x_train = X(train_idx, :);
y_train = Y(train_idx);
% train/predict with niave bayes classification
y_pred = NB(x_train, y_train, x_test);
% calculate classification accuracy
acc = sum(y_pred == y_test)/length(y_test);
fprintf('\n PASSIVE LEARNING classification accuracy: %.2f %%', acc);

%% DEMO 2a:
% RANDOM SAMPLE TRAINING: passive/supervised learning for an increasing 
% label budget 

% parameters
T = 30; % maximum number of runs (max label budget T*B = 600)
B = 20; % batch size
reps = 20; % number of repeats for each experiment

% error
e_rs = [];
for t = 1:T
    % verbose
    fprintf('\nQUERY BUDGET ------ %d', t*B)
    % accuracy for each repeat
    acc = [];
    for r = 1:reps
        % DIVIDE DATA INTO: 
        % TRAINING DATA (x_train/y_train)
        % TEST DATA (x_train/y_train)
        % use random indices to define test-set
        [idx,~,test_idx] = dividerand(length(data),2/3,0,1/3);
        % available data
        X = data(idx, 1:end-1); Y = data(idx, end);
        % test-set
        x_test = data(test_idx, 1:end-1); y_test = data(test_idx, end);
        % define the training-set by a random sample from availaible data
        train_idx = randperm(size(X,1), t*B);
        x_train = X(train_idx, :);
        y_train = Y(train_idx);
        
        % CLASSIFICATION
        y_pred = NB(x_train, y_train, x_test);
        % record accuracy of prediction
        acc = [acc, sum(y_pred == y_test)/length(y_test)]; %#ok<AGROW>
    end
    e_rs = [e_rs, 1-mean(acc)]; %#ok<AGROW>
end


%% DEMO 2b:
% CLUSTER BASED ACTIVE LEARNING: the DH learner for an increasing label budget

e_al = []; 
for t = 1:T
    % verbose
    acc= [];
    fprintf('\nQUERY BUDGET ------ %d', t*B)
    for r = 1:reps 
        % DIVIDE DATA INTO: 
        % AVAILABLE DATA (X)
        % HIDDEN LABELS (Y)
        % TEST DATA (x_test/y_test)
        % use random indices to define test-set
        [idx,~,test_idx] = dividerand(length(data),2/3,0,1/3);
        % available data
        X = data(idx, 1:end-1); Y = data(idx, end);
        % test-set
        x_test = data(test_idx, 1:end-1); y_test = data(test_idx, end);
        
        % ------------- INITIAL CLUSTERING -------------- % 
        [u, ch] = h_cluster(X, 'max_clusters', 250);
        
        % -- HIERARCHICAL SAMPLING FOR ACTIVE LEARNING -- %
        % -- the DH active learning algorithm
        [xl, z, p, l] = DH_AL(u, ch, B, t, Y);
        % define training-set (inputs/outputs).
        % Note, first column of xl is input indices, second is class labels
        train_idx = xl(:, 1);
        x_train  = X(train_idx, :);
        y_train = xl(:, 2);
        
        % --------------- CLASSIFICATION ---------------- %       
        y_pred = NB(x_train, y_train, x_test);
        % record accuracy of prediction
        acc = [acc, sum(y_pred == y_test)/length(y_test)]; %#ok<AGROW>
    end
    e_al = [e_al, 1 - mean(acc)];    %#ok<AGROW>
end

% PLOT
figure(3) 
plot(B:B:B*T, e_rs, '--');
hold on
plot(B:B:B*T, e_al, '--');
xlabel('label budget (n)')
ylabel('classification error (e)')
legend('passive learning','active learning')
hold off 
