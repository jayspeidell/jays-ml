import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

class MultiLabel(): 
    def __init__(self, base_model, labels):
        self.base_model = base_model
        self.labels = labels
        self.num_labels = len(labels)
        self.models = []
        self.generate_copies()
        self.scores = []

        self.initialize()
        
    def generate_copies(self):
        for _ in range(self.num_labels):
            self.models.append(clone(self.base_model))
    
    def initialize(self):
        print('Multi-Label Object Initialized')
        print(('%d Labels: "' + '", "'.join(self.labels) +'"') % self.num_labels)
        print(self.base_model)
    
    def cm_heatmap(self, arr, title):
        plt.figure('cm_heatmap')
        plt.title(title + ' confusion matrix')
        sns.heatmap(arr, square=True, annot=True, cmap='YlOrRd', fmt='g')
        plt.show()

    def scoring(self, y_test, y_pred, i):
        cm = confusion_matrix(y_test, y_pred)
        precision = cm[1][1] / (cm[1][1] + cm[0][1])
        print('Precision: %.3f' % precision)
        recall = cm[1][1] / (cm[1][1] + cm[1][0])
        print('Recall: %.3f' % recall)
        f1_score = fbeta_score(y_test, y_pred, beta=1)
        print('F1 score: ' + str(f1_score))
        self.cm_heatmap(cm, self.labels[i])
        self.scores.append([self.labels[i], precision, recall, f1_score])
        print()  
    
    def fit(self, X, y, X_test, y_test):
        for i, model in enumerate(self.models):
            print('Fitting model %d for label %s.' % (i+1, self.labels[i]))
            model.fit(X, y[:,i])
            print('Saving model.')
            self.models[i] = model
        print('Training complete.')
    
    def fit_test(self, X, y, X_test, y_test):
        for i, model in enumerate(self.models):
            print('Fitting model %d for label %s.' % (i+1, self.labels[i]))
            model.fit(X, y[:,i])
            print('Saving model.')
            self.models[i] = model
            self.scoring(y_test[:,i], model.predict(X_test), i)
        print('Training complete.')
    
    def test(self, X_test, y_test):
        for i, model in enumarate(self.models):
            print('Testing model %d for label %s.' % (i+1, self.labels[i]))
            self.scoring(y_test[:,i], model.predict(X_test), i)
        print('Testing complete')
            
        
    def get_scores(self):
        if len(self.scores) > 0:
            for score in self.scores:
                print('Scores for label "' + str(score[0]) + '".')
                print('Precision: %.3f' % score[1])
                print('Recall: %.3f' % score[2])
                print('F1 Score: %.3f\n' % score[3])
        else:
            print('No score data. Scoring data is saved after model testing.')   

def stack_predictions(X_train, y_train, X_test, submit, K, *models):
    train_preds = pd.DataFrame(index=np.array(range(X_train.shape[0])))
    test_preds = pd.DataFrame(index=np.array(range(X_test.shape[0])))
    submit_preds = pd.DataFrame(index=np.array(range(submit.shape[0])))
    folds = KFold(n_splits=K, shuffle=True)
    
    fold_n = 0
    train_folds = np.zeros(len(train_preds))
    for train_index, test_index in folds.split(X_train):
        train_folds[test_index] = fold_n
        fold_n += 1
    
    fold_n = 0
    test_folds = np.zeros(len(test_preds))
    for train_index, test_index in folds.split(X_test):
        test_folds[test_index] = fold_n
        fold_n += 1
    
    fold_n = 0
    submit_folds = np.zeros(len(submit_preds))
    for train_index, test_index in folds.split(submit):
        submit_folds[test_index] = fold_n
        fold_n += 1
    
    for m, model in enumerate(models):
        print('Selecting model %d.' % (m+1))
        col = 'pred_col_' + str(m)
        train_preds[col] = np.nan
        test_preds[col] = np.nan
        submit_preds[col] = np.nan
        
        for fold in range(K):
            print('Processing a fold...')
            current_model = clone(model)
            current_model.fit(X_train[np.where(train_folds!=fold)], y_train[np.where(train_folds!=fold)])
            
            train_preds[col].iloc[np.where(train_folds==fold)] = current_model.predict(
                X_train[np.where(train_folds==fold)])
            
            test_preds[col].iloc[np.where(test_folds==fold)] = current_model.predict(
                X_test[np.where(test_folds==fold)])
            
            submit_preds[col].iloc[np.where(submit_folds==fold)] = current_model.predict(
                submit[np.where(submit_folds==fold)])  

    return train_preds, test_preds, submit_preds
