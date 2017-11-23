from datetime import datetime
import utils
from visit_models import experiment_dict as visit
from rating_models import experiment_dict as rating

from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import seaborn as sb

def run(args):

    task = args['model']
    submit = args['submission']

    # 1.)Load data for training model
    X_train_full, y_train_full = utils.load_train_data(task)

    if submit:
        # making a submission; train on all given data
        print('fitting models to entire training set')
        X_train, y_train = X_train_full, y_train_full
        X_test = utils.load_test_data(task)
    else:
        # running an experiment - cross validate with train/test split
        test_size = args['test_size']
        print('fitting models to cv train/test split with train% = {}'.format(1-test_size))
        X_train, X_val, y_train, y_val = train_test_split(X_train_full,y_train_full, test_size=test_size, random_state=args['random_state'])


    # 2.) Get pipeline
    if task == 'Visit':
        pipeline_detail = visit[args['expt']]
        X_train, y_train = utils.sample_negatives(X_train, y_train, 2)
        if not submit:
            X_val, y_val = utils.sample_negatives(X_val, y_val, 1)
    else:
        pipeline_detail = rating[args['expt']]

    pipeline = pipeline_detail['pl']


    # Fit model to training data
    print('fitting model to array sizes (xtrain, ytrain)={}'.format([i.shape for i in [X_train, y_train]]))
    print('fitting experiment pipeline with signature={}'.format(pipeline))

    pipeline.fit(X_train, y_train)

    # 3.) For non-submission experiments, get the best parameters from grid search
    if submit:
        fname_spec = '_submission_'
    else:
        # log all results + call out the winner
        if hasattr(pipeline, 'best_params_'):
            print('best gridsearch score={}'.format(pipeline.best_score_))
            print('best set of pipeline params={}'.format(pipeline.best_params_))
            print('now displaying all pipeline param scores...')
            cv_results = pipeline.cv_results_
            for params, mean_score, scores in list(zip(cv_results['params'], cv_results['mean_test_score'], cv_results['std_test_score'])):
                print("{:0.3f} (+/-{:0.03f}) for {}".format(mean_score, scores.std() * 2, params))
        fname_spec = '_expt_'

    model_name = utils.short_name(pipeline) + fname_spec + datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')


    # 4.) Prepare submission
    if submit:
        print('writing predictions to formatted submission file')
        predictions = pipeline.predict(X_test)
        if hasattr(pipeline, 'best_params_'):
            print('predicting test values with best-choice gridsearch params')
        utils.create_submission(predictions, pipeline_detail['name'], X_test)
    else:
        cv = args['k-fold']
        print('cross validating model predictions with cv={}'.format(cv))
        predictions = cross_val_predict(pipeline, X_val, y_val, cv=cv)

        # print("cross val prediction", accuracy_score(y_val, predictions))
        print("cross val prediction", mean_squared_error(y_val, predictions))

        predictions_train = pipeline.predict(X_train)
        predictions_test = pipeline.predict(X_val)

        if task == 'Visit':
            print('obtained train accuracy = {:.2f}, test accuracy = {:.2f}  pipeline={} '.format(
                accuracy_score(y_train, predictions_train),
                accuracy_score(y_val, predictions_test),
                pipeline))

            print('calculating confusion matrix')
            try:
                cf = confusion_matrix(y_val, predictions)
                print("confusion matrix: ", cf)
                sb.heatmap(cf)
            except RuntimeError as e:
                print('plotting error. matplotlib backend may need to be changed (see readme). error={}'.format(e))
                print('plot may still have been saved, and model has already been saved to disk.')
        else:
            print('obtained train mse = {:.2f} test mse={}, pipeline={} '.format(
                mean_squared_error(y_train, predictions_train),
                mean_squared_error(y_val, predictions_test),
                pipeline))

        if args['cross_val_score']:
            # this gives a better idea of uncertainty, but it adds 'cv' more
            print('cross validating model accuracy with cv={}'.format(cv))
            scores = cross_val_score(pipeline, X_val, y_val, cv=cv)
            print('obtained accuracy={:0.2f}% +/- {:0.2f} with cv={}, \
                                        pipeline={} '.format(scores.mean() * 100,
                                                             scores.std() * 100 * 2,
                                                             cv,
                                                             pipeline))



    print('completed with pipeline {}'.format(pipeline_detail['name']))




