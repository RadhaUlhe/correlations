import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def get_anomaly_correlation(result):

    result_df = pd.DataFrame(result.tolist())
    result_df = result_df.dropna(how='any', subset=['anomalyAnalysis'])
    result_df.reset_index(drop="Index", inplace=True)
    res_df = pd.DataFrame({'metrics': result})
    result_temp_df = pd.DataFrame()
    result_temp_df['metric'] = res_df['metrics'].apply(lambda x: x.get('metricName', 0))
    result_temp_df['anomaly'] = res_df['metrics'].apply(lambda x: x.get('anomalyAnalysis', 0))
    result_temp_df = result_temp_df[result_temp_df.anomaly != 0]
    result_temp_df.reset_index(drop="Index", inplace=True)

    # building a dataframe for anomaly scores
    df = pd.DataFrame()
    idx = 0
    for row in result_temp_df.itertuples():
        if idx == 0:
            # get scores list for each metric in each timebucket
            scores = []
            anomaly_dict = getattr(row, 'anomaly')
            for key, value in anomaly_dict.items():
                scores.append(anomaly_dict[key].get('anomalyscore'))
            df = pd.DataFrame(scores, columns=[getattr(row, 'metric')])
        else:
            # get scores list for each metric in each timebucket
            scores = []
            anomaly_dict = getattr(row, 'anomaly')
            for key, value in anomaly_dict.items():
                scores.append(anomaly_dict[key].get('anomalyscore'))
            temp_df = pd.DataFrame(scores, columns=[getattr(row, 'metric')])
            df = pd.concat([df, temp_df], ignore_index=False, axis=1, sort=False)
        idx = idx + 1

    # correlation among metrics having anomalies in same time bucket
    col_list = df.columns
    corr_list = []
    for idx, row in df.iterrows():
        bucket_corr_list = []
        for col in range(0, len(df.columns) - 1):
            x = df.iloc[idx, col]
            if x == 0:
                bucket_corr_list.append([col_list[col]])
        flat_list = [item for sublist in bucket_corr_list for item in sublist]
        corr_list.append(list(set(flat_list)))

    metric_names = []
    metric_values = []
    for row in result_df.itertuples():
        metric_names.append(getattr(row, 'metricName'))
        metric_values.append(getattr(row, 'version2')['data']['y'].to_list())
    corr_df = pd.DataFrame(metric_values)
    corr_df = corr_df.transpose()
    corr_df.columns = metric_names
    corr_df = corr_df.loc[:, ~corr_df.columns.duplicated()]

    # forming 'time-buckets' object for final json
    result_df['time_buckets'] = None
    idx = 0
    for row in result_df.itertuples():
        time_buckets = {}
        is_anomaly = False
        correlations = {}
        anomaly_dict = getattr(row, 'anomalyAnalysis')
        metric_name = getattr(row, 'metricName')
        for key, value in anomaly_dict.items():
            score = anomaly_dict[key].get('anomalyscore')
            percentage_score = anomaly_dict[key].get('percentagescore')
            hypothesis_score = anomaly_dict[key].get('hypothesisscore')
            quantile_score = anomaly_dict[key].get('quantilescore')
            starttime = anomaly_dict[key].get('starttime')
            endtime = anomaly_dict[key].get('endtime')
            aggr = anomaly_dict[key].get('aggregator')
            timeinterval = anomaly_dict[key].get('timeinterval')

            if score == 0:
                metrics = {}
                is_anomaly = True
                correlations_list = list(np.setdiff1d(corr_list[key], getattr(row, 'metricName')))
                for metric in correlations_list:
                    for row_temp in result_df.itertuples():
                        if metric == getattr(row_temp, 'metricName'):
                            data_1 = corr_df[metric_name].to_list()
                            data_2 = corr_df[metric].to_list()
                            corr, _ = pearsonr(data_1, data_2)
                            if np.isnan(corr):
                                corr = 0.0
                            # fetch metric data
                            if abs(corr) > 0.5:
                                v2_data = getattr(row_temp, 'version2')
                                v2_current_bucket_data = v2_data['data'][(v2_data['data']['x'] >= starttime) &
                                                                         (v2_data['data']['x'] <= endtime)]
                                metrics[metric] = {"data": v2_current_bucket_data,
                                                   "corrvalue": corr}

                correlations = {"corrnum": len(metrics),
                                "metrics": metrics}

                time_buckets[key] = {"timeinterval": timeinterval,
                                     "starttime": starttime,
                                     "endtime": endtime,
                                     "aggregator": aggr,
                                     "anomalyscore": score,
                                     "percentagescore": percentage_score,
                                     "hypothesisscore": hypothesis_score,
                                     "quantilescore": quantile_score,
                                     "correlations": correlations}

            else:
                is_anomaly = False
                correlations = {}
                time_buckets[key] = {"timeinterval": timeinterval,
                                     "starttime": starttime,
                                     "endtime": endtime,
                                     "aggregator": aggr,
                                     "anomalyscore": score,
                                     "percentagescore": percentage_score,
                                     "hypothesisscore": hypothesis_score,
                                     "quantilescore": quantile_score,
                                     "correlations": correlations}

        result_df.loc[idx, 'time_buckets'] = [time_buckets]
        idx = idx + 1

    result_df.drop(columns=['anomalyAnalysis'], inplace=True)
    result_df = result_df.rename(columns={'time_buckets': 'anomalyAnalysis'})
    return result_df
