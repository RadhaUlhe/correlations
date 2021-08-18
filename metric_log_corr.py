import os
import time
import pandas as pd
from extract_json_data import get_data
import analyse_collinear_clusters
from build_model import Regression
import numpy as np


def get_target_cluster(log_features_df, original_log_features_df, sorted_log_features_df, log_analysis_df, outlier_list):

    temp_df = pd.DataFrame(columns=['target_cluster', 'target_cluster_evidence', 'rare_cluster', 'rare_cluster_evidence', 'freq_cluster', 'freq_cluster_evidence'])
    appended_list = []
    target_list = []
    rare_list = []
    for idx, row in sorted_log_features_df.iterrows():
        lst1 = []
        lst2 = []
        lst3 = []
        lst4 = []
        lst5 = []
        lst6 = []
        lst7 = []
        lst8 = []
        t1 = tuple()
        t2 = tuple()
        t = tuple()
        for indx in range(0, sorted_log_features_df.shape[1]):

            if original_log_features_df.loc[idx, row[indx]] in outlier_list[row[indx]]:
                outlier = "outlier"
            else:
                outlier = "not-outlier"

            if log_features_df.loc[idx, row[indx]] == 0:
                continue
            if log_analysis_df.loc[row[indx], 'rareCluster'] == 'yes':
                if row[indx] not in appended_list:
                    lst1.append([row[indx], original_log_features_df.loc[idx, row[indx]], outlier])
                    lst2.append(log_analysis_df.loc[row[indx], 'combineClust'])
                    lst7.append(original_log_features_df.loc[idx, row[indx]])

                else:
                    lst5.append([row[indx], original_log_features_df.loc[idx, row[indx]], outlier])
                    lst6.append(log_analysis_df.loc[row[indx], 'combineClust'])
                    lst8.append(original_log_features_df.loc[idx, row[indx]])

                appended_list.append(row[indx])

            else:
                lst3.append([row[indx], original_log_features_df.loc[idx, row[indx]], outlier])
                lst4.append("None")

        target_list.append(lst7)
        rare_list.append(lst8)
        temp_df.at[idx, 'target_cluster'] = lst1
        temp_df.at[idx, 'target_cluster_evidence'] = lst2

        temp_df.at[idx, 'rare_cluster'] = lst5
        temp_df.at[idx, 'rare_cluster_evidence'] = lst6

        temp_df.at[idx, 'freq_cluster'] = lst3
        temp_df.at[idx, 'freq_cluster_evidence'] = lst4

    return temp_df, target_list, rare_list


def get_features_target(target_metric_dict, features_df, clusters_df, result_dir):

    # get metric data
    data_dict = target_metric_dict['version2']['data']
    metric_df = pd.DataFrame.from_dict(data_dict, orient='index')
    metric_df.rename(columns={"x": "TimeBucket", "y": "Value"}, inplace=True)
    metric_df.TimeBucket = metric_df.TimeBucket.astype(int)

    # get log data
    features_df = features_df.fillna(0)
    features_df.TimeBucket = features_df.TimeBucket.astype(float).astype(int)

    # Find correlation between clusters and reduce multicollinearity
    temp_df = features_df.copy()
    temp_df = temp_df.drop(columns=['TimeBucket'], axis=1)
    corr_matrix = temp_df.corr(method="pearson")

    cl = analyse_collinear_clusters.Analysis(corr_matrix, clusters_df)
    analysis_result_df = cl.run_analysis()
    analysis_result_df = analysis_result_df.reset_index(drop="index")
    analysis_result_df.to_csv(os.path.join(result_dir, "fulldata.csv"))

    # remove collinear clusters
    sequence_clusters_list = analysis_result_df['sequenceClusterNo'].tolist()
    final_features_df = features_df[sequence_clusters_list]
    final_features_df.columns = range(0, final_features_df.shape[1])
    final_features_df.loc[:, 'TimeBucket'] = features_df.loc[:, 'TimeBucket'].values

    # merge two dataframes : log and metric
    features_metric_data = metric_df.merge(final_features_df, on="TimeBucket")
    if len(features_metric_data) == 0:
        print("No match in time buckets of logs and metrics")
        exit()
    features_metric_data.to_csv(os.path.join(result_dir, "intermediateResult.csv"))

    return features_metric_data, analysis_result_df


def get_rare_cluster(row, max_weight):

    if row['idfWeight'] > 0.4 * max_weight:
        return 'yes'
    else:
        return 'no'


def detect_outlier(data_1, threshold):
    outliers = []
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)

    for y in data_1:
        z_score = (y - mean_1) / std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)

    return list(set(outliers))


def metric_log_corr(base_dir, target_metric):
    # set path for output
    result_dir = os.path.join(base_dir, "correlation")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    # get data from logs and metrics
    target_metric_dict, features_df, clusters_df = get_data(base_dir, target_metric)

    # get APM metric and log matrix in one dataframe
    features_target_data, log_analysis_df = get_features_target(target_metric_dict, features_df, clusters_df, result_dir)

    # build model
    lr = Regression(features_target_data)

    # find correlation weight of each sequence cluster
    corr_weight_list = lr.get_corr_weight()

    # find the sequence cluster weight based on correlation weight and no. of occurrences of sequence cluster
    weighted_data, final_weight_list, idf_weight_list = lr.get_seqclust_weight(corr_weight_list)

    # get maximum impact clusters (sort in descending order)
    log_metric_df = features_target_data.copy()
    log_features_df = log_metric_df.drop(["TimeBucket", "Value"], axis=1)
    log_features_df = log_features_df.fillna(0)
    original_log_features_df = log_features_df.copy()
    metric_df = features_target_data['Value'].values

    temp_df = log_features_df.copy()
    temp_df[temp_df != 0] = 1
    log_features_df = np.multiply(temp_df, corr_weight_list)

    ################## Get Rare cluster ############################################
    log_analysis_df.reset_index(drop="Index", inplace=True)
    log_analysis_df['idfWeight'] = idf_weight_list
    log_analysis_df['corrWeight'] = corr_weight_list
    maxWeight = max(idf_weight_list)
    log_analysis_df['rareCluster'] = log_analysis_df.apply(get_rare_cluster, args=(maxWeight,), axis=1)

    ############# Detect outlier from cluster frequency ###########################
    outlier_list = []
    for column in original_log_features_df:
        outlier_list.append(detect_outlier(original_log_features_df.loc[:, column], 2))


    ################### ERROR/WARN ##################################################

    # get accurate error/warn clusters responsible for change in APM value
    # get all error/warn clusters with weights
    selected_topic_list = ["ERROR", "CRITICAL ERROR", "WARN"]
    error_warn_clusters_df = log_analysis_df.loc[(log_analysis_df['topic'] == 'ERROR') | (log_analysis_df['topic'] == 'WARN') | (log_analysis_df['topic'] == 'CRITICAL ERROR')]
    error_warn_clusters_df_selected = error_warn_clusters_df[["topic", "idfWeight", "rareCluster", "corrWeight"]]

    # select only error/warn clusters from df
    lst = list(error_warn_clusters_df.index)
    error_warn_log_features_df = log_features_df[lst]
    original_error_warn_log_features_df = original_log_features_df[lst]
    #error_warn_log_df.to_csv(os.path.join(resultDir, "errorpriority.csv"))
    sorted_error_warn_log_features_df = pd.DataFrame(
        data=error_warn_log_features_df.columns.values[np.argsort(-error_warn_log_features_df.values, axis=1)],
        columns=['tag_{}'.format(i) for i in range(error_warn_log_features_df.shape[1])])
    result_df, target_list, rare_list = get_target_cluster(error_warn_log_features_df, original_error_warn_log_features_df, sorted_error_warn_log_features_df,
                                                log_analysis_df, outlier_list)

    ####################################################################################

    log_analysis_df = pd.concat([features_target_data['TimeBucket'], features_target_data['Value'], result_df], axis=1)
    log_analysis_df.to_csv(os.path.join(result_dir, "analysisResult.csv"))

    print("success")



base_dir = "/home/dell/Desktop/metric_log_corr/datascience/data/417/"
target_metric = "avg:trace.servlet.request.hits{resource_name:get_/dogcount,variable1}.as_count()"
metric_log_corr(base_dir, target_metric)
