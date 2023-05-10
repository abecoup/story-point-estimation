from lda_clustering import *
import sys
import os

def main():
    if len(sys.argv) != 6:
        print("Exactly five arguments must be supplied: path to data directory, path to LDA model (.model), path to save results to, evaluation method (one of 'MAE', 'MdAE', or 'sil'), and LHC variant ('LHC-SE' or 'LHC-TC-SE')")
        print("Usage: python run_sp_estimation.py [PATH TO DATASET] [PATH TO LDA MODEL] [PATH TO SAVE RESULTS TO] [Cluster Building Strategy:'MAE', 'MdAE', or 'sil'] [Algorithm Variant: 'LHC-SE' or 'LHC-TC-SE']")
        sys.exit(1)

    # Get command line args
    data_path = sys.argv[1]
    lda_path = sys.argv[2]
    result_path = sys.argv[3]
    ev = sys.argv[4]
    variant = sys.argv[5]

    # If results directory does not exist, create it
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Load LDA model
    print("Loading LDA model...")
    lda_model = LdaModel.load(lda_path)

    for project_name in get_project_names(data_path):
        train_data = load_project_data(data_path, "train", project_name, variant)
        valid_data = load_project_data(data_path, "valid", project_name, variant)
        test_data = load_project_data(data_path, "test", project_name, variant)

        # Fitting LDA model to training, testing and validation data
        dtm_lda = extract_topic_distributions(train_data, valid_data, test_data, lda_model)

        # grab extra features
        if variant == "LHC-TC-SE":
            train_extra = train_data.drop(columns=['issuekey', 'storypoint', 'issue_context', 'project'])
            train_extra = pd.DataFrame(np.reshape(train_extra.values, (train_extra.shape[0], -1)))
            assert np.isnan(train_extra.values.astype(np.float64)).sum() == 0, "There are missing values in the data"

            valid_extra = valid_data.drop(columns=['issuekey', 'storypoint', 'issue_context', 'project'])
            valid_extra = pd.DataFrame(np.reshape(valid_extra.values, (valid_extra.shape[0], -1)))
            assert np.isnan(valid_extra.values.astype(np.float64)).sum() == 0, "There are missing values in the data"

            test_extra = test_data.drop(columns=['issuekey', 'storypoint', 'issue_context', 'project'])
            test_extra = pd.DataFrame(np.reshape(test_extra.values, (test_extra.shape[0], -1)))

        # Merge fitted data with extra features if LHC-TC-SE
        dtm = {}
        if variant == "LHC-TC-SE":
            dtm['train'] = pd.concat([pd.DataFrame(dtm_lda['train']), train_extra], axis=1)
            dtm['valid'] = pd.concat([pd.DataFrame(dtm_lda['valid']), valid_extra], axis=1)
            dtm['test'] = pd.concat([pd.DataFrame(dtm_lda['test']), test_extra], axis=1)
        else: # LHC-SE
            dtm['train'] = pd.DataFrame(dtm_lda['train'])
            dtm['valid'] = pd.DataFrame(dtm_lda['valid'])
            dtm['test'] = pd.DataFrame(dtm_lda['test'])
        
        assert dtm['train'].shape[1] == dtm['valid'].shape[1] == dtm['test'].shape[1], "The number of columns in train, valid, and test are not equal"

        # perform clustering
        train_data['labels'] = perform_clustering(train_data, test_data, valid_data, dtm,
                    FE = "LDA",
                    verbose = True,
                    project_name = project_name,
                    ev = ev,
                    lda_model = lda_model,
                    result_dir = result_path)

        # find statistics per cluster
        val_data = validate(data=train_data, test=test_data, dtm_train=dtm['train'], dtm_test=dtm['test'])

        results = val_data['results']

        # Save estimations
        results.to_csv(result_path + project_name + '_results.csv', index=False)

        # save_project_metrics(project_name, mae=val_data['mae_mdae'][0], mdae=val_data['mae_mdae'][1], variant)

        # Print estimation statistics
        ae_sp_closest = abs(results['sp'] - results['closest_sp'])
        print("\nStory Point - Absolute Error when matching with closest point:\n")
        # print(ae_sp_closest.describe(include= 'all'))
        print("\nMean of Absolute Error: ", ae_sp_closest.mean())
        print("Median of Absolute Error: ", ae_sp_closest.median())

        ae_sp_cluster_mean = abs(results['sp'] - results['mean_cluster_sp'])
        print("\nStory Point - Absolute Error when matching with cluster mean:\n")
        # print(ae_sp_cluster_mean.describe(include= 'all'))
        print("\nMean of Absolute Error: ", ae_sp_cluster_mean.mean())
        print("Median of Absolute Error: ", ae_sp_cluster_mean.median())

        ae_sp_cluster_median = abs(results['sp'] - results['median_cluster_sp'])
        print("\nStory Point - Absolute Error when matching with cluster median:\n")
        # print(ae_sp_cluster_median.describe(include= 'all'))
        print("\nMean of Absolute Error: ", ae_sp_cluster_median.mean())
        print("Median of Absolute Error: ", ae_sp_cluster_median.median())

        print("\n########################################################################\n")
    

if __name__ == '__main__':
    main()