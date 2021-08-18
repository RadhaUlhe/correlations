import json
import os
import pandas as pd
import numpy as np
import dateutil.parser as dp
import logging
logger = logging.getLogger('dslogger')


def fetch_log_feature_metric(result_df, serviceName):
    selected_df = pd.DataFrame()
    features_df = pd.DataFrame()
    if len(result_df) != 0:
        for key in result_df['data'][0].keys():
            if key == "version2" or key == "v1v2":
                clusters_df = pd.DataFrame(result_df['data'][0][key]['clusters'])   # version2 & v1v2
                if len(clusters_df) != 0:
                    if key == "version2":
                        identity = "_v2"
                    else:
                        identity = "_v1v2"
                    clusters_df['uniqueid'] = serviceName + "/" + clusters_df['id'].astype(str) + identity

                    clusters_df = clusters_df.loc[clusters_df["topic"].isin(["ERROR", "WARN", "CRITICAL ERROR"])]
                    if len(clusters_df) == 0:
                        continue
                    clusters_df = clusters_df.reset_index(drop="index")

                    ## Form dataframe out of given json with cluster id
                    i = 0
                    clustL = []
                    for j, cluster in clusters_df.iterrows():
                        # for cluster in clustersL:
                        for k in cluster['TimeBuckets']:
                            t = k
                            count = cluster['TimeBuckets'][k]
                            clustL.append(
                                {"cluster": cluster['uniqueid'], "message": cluster["topic"] + " " + cluster["combineClust"], "time": t,
                                "count": count})
                        i = i + 1
                        selected_df = pd.DataFrame(clustL)

                    ## Create group of clusters with same time buckets
                    similar_timestamp_group = selected_df.groupby("time")
                    df_list = [group for _, group in similar_timestamp_group]
                    df_list_length = len(df_list)
                    ## creating feature matrix for input to model
                    final_df = pd.DataFrame(index=np.arange(df_list_length))
                    similar_cluster_group = pd.DataFrame()
                    i = 0
                    for df in df_list:
                        similar_cluster_group = df.groupby("cluster", as_index=False).agg({'count': 'sum'})
                        final_df.at[i, 'TimeBucket'] = df.iloc[[0]]['time'].values[0]
                        for idx, row in similar_cluster_group.iterrows():
                            final_df.at[i, row['cluster']] = row['count']
                        i = i + 1
                    final_df.fillna(0, inplace=True)
                    #print("log feature matrix formed!")
                    #final_df.to_csv(os.path.join(result_dir, "logFeature.csv"))
                    #clusters_df.to_csv(os.path.join(result_dir, "logData.csv"))
                    if key == "version2":
                        features_df = final_df
                    elif key == "v1v2" and len(features_df) != 0:
                        features_df = features_df.merge(final_df, on="TimeBucket", how="outer")
                    elif key == "v1v2" and len(features_df) == 0:
                        features_df = final_df
                    features_df.fillna(0, inplace=True)
                else:   # Continue if clusters are not present in "version2" or "v1v2"
                    continue
            else:  # Continue if key is other than "version2" or "v1v2"
                continue
        if len(features_df) == 0:
            #print("log-log correlation error: Version2 and v1v2 clusters are not present in given .json")
            logging.error(f"log-log correlation error: Version2 and v1v2 clusters are not present in given result .json of service:{serviceName}")
            return pd.DataFrame()
        else:
            return features_df
    else:
        #print("log-log correlation error : Empty result .json file")
        logging.error(f"log-log correlation error : Empty result .json file for service:{serviceName}")
        return pd.DataFrame()


def get_unix_timestamp(timestamp):
    parsed_t = dp.parse(timestamp)
    t_in_seconds = int(parsed_t.strftime('%s')) * 1000
    timestamp = str(t_in_seconds)
    return timestamp


def run_log_log_corr(canaryDir, service_id, cluster_id, timeinterval_start, timeinterval_end):
    features_df = pd.DataFrame()
    if not (os.path.isdir(canaryDir)):
        logging.critical(f"Canary:{canaryDir} does not exist")
        raise (Exception(f"Canary:{canaryDir} does not exist"))
    logging.info(f"Received canary directory path for log-log correlation :{canaryDir}")
    # Check for no. of services in application
    serviceName = [name for name in os.listdir(canaryDir) if
                    os.path.isdir(os.path.join(canaryDir, name)) and name not in "result"]
    service_count = 1
    for service in serviceName:
        servicePath = service + "/"
        logDir = os.path.join(canaryDir, servicePath)
        resultJsonPath = os.path.join(logDir, 'result', "logAnalysisResult.json")

        if not os.path.exists(resultJsonPath):
            logging.critical(f"Loganalysis Result .json file is missing in Canary: {canaryDir}:{service}")
            continue
        else:
            with open(resultJsonPath, "r") as file:
                resultJson = json.load(file)
                file.close()
            logging.info(f"Loaded both {resultJsonPath}")

            # Get feature matrix of a service
            service_features_df = fetch_log_feature_metric(resultJson, service)

            # Merge feature matrices of services
            if len(service_features_df) != 0:
                logging.info(f"Features matrix is formed for service:{service}")
                if service_count == 1:
                    features_df = service_features_df
                elif service_count > 1:
                    features_df = features_df.merge(service_features_df, on="TimeBucket", how="outer")
                features_df.fillna(0, inplace=True)
                service_count = service_count + 1
            else:
                continue

    if len(features_df) != 0:
        # Create correlation matrix
        features_wo_timebucket = features_df.loc[:, features_df.columns != "TimeBucket"]
        corr_matrix = features_wo_timebucket.corr(method="spearman")

        # Fetch correlation of selected service cluster with other clusters in selected time interval
        #start = get_unix_timestamp(timeinterval_start)
        #end = get_unix_timestamp(timeinterval_end)
        features_df.set_index("TimeBucket", inplace=True)
        features_df.sort_index(axis=0, inplace=True)
        #features_df.to_csv(os.path.join(canaryDir, "features.csv"))
        selected_interval = features_df.loc[timeinterval_start:timeinterval_end].sum()
        result_df = pd.DataFrame()
        i = 0
        # Create final .json object with clusters having occurrences in time interval with respective correlation coefficients
        for idx, items in selected_interval.iteritems():
            if idx in corr_matrix[service_id + "/" + cluster_id] and idx != service_id + "/" + cluster_id and items != 0:
                service = idx.split("/")[0]
                cluster = idx.split("/")[1]
                result_df.loc[i, 'service_id'] = service
                result_df.loc[i, 'cluster_id'] = cluster
                result_df.loc[i, 'occurrences'] = items
                result_df.loc[i, 'correlation_coefficient'] = corr_matrix[service_id + "/" + cluster_id][idx]
                i = i+1
        result_df.sort_values(by="correlation_coefficient", ascending=False, inplace=True)
        result_df.reset_index(drop="Index", inplace=True)
        result_dict = result_df.to_dict(orient="records")
        return json.dumps(result_dict)
    else:
        logging.critical(f"No data found for logs analysis result in Canary: {canaryDir}")
        raise (Exception(f"No data found for logs analysis result in Canary:{canaryDir}"))


