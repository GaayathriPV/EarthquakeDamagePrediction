import matplotlib.pyplot as plt
from numpy.core.fromnumeric import reshape
import pandas as pd
import numpy as np
import seaborn as sns
import math

class DataManager:
    def __init__(self) -> None:
        pass

    def extract_numerical_data(df:pd.DataFrame, df_analysed:pd.DataFrame):
        return pd.DataFrame(df[df_analysed.query('(Scale == "ratio") or (Scale == "interval")').index])

    def extract_categorical_data(df:pd.DataFrame, df_analysed:pd.DataFrame):
        return pd.DataFrame(df[df_analysed.query('(Scale == "nominal") or (Scale == "ordinal")').index]).astype('category')

    def data_analyser(df:pd.DataFrame, scale_type):
        analyze_res = {}
        for attr in df.columns:
            temp = df[str(attr)]
            distinct = temp.unique()
            count = temp.size
            missing_value = temp.isnull().sum()
            analyze_res[str(attr)] = {
                'Cardinality (#)' : len(distinct),
                'Miss (#)' : missing_value,
                'Miss (%)' : missing_value/count * 100.0,
                'Domain' : distinct,
                'Stored_data_type' : temp.dtype
                }

        df_analyze = pd.DataFrame.from_dict(analyze_res,orient='index')
        df_analyze.index.name='Feature'
        df_analyze.insert(loc=1,column='Scale',value=scale_type)
        return ({'df':df_analyze, 'instances_number':df.index.size, 'features_number':df.columns.size})

    def numerical_report(df:pd.DataFrame, outlier_max_dist, out_method='qrt')-> pd.DataFrame:
        rep_num = {}
        for attr in df.columns:
            temp = df[attr]
            distinct = temp.unique()
            count = temp.size
            missing_value = temp.isnull().sum()
            min_val = temp.min()
            max_val = temp.max()
            mean_val = temp.mean()
            std_val = temp.std()
            qr1 = temp.quantile(0.25)
            qr2 = temp.quantile(0.50)
            qr3 = temp.quantile(0.75)
            iqr = qr3 - qr1

            # outlier boundaries
            out_upper = 0
            out_lower = 0
            # method 1
            if(out_method=='qrt'):
                out_lower = qr1 - iqr * outlier_max_dist
                out_upper = qr3 + iqr * outlier_max_dist
            # method 2
            elif(out_method=='std'):
                out_lower = mean_val - std_val * outlier_max_dist
                out_upper = mean_val + std_val * outlier_max_dist
            else:
                raise Exception(str('Not defined method: {}'.format(out_method)))

            outliers_num = temp.where(temp>out_upper).count() + temp.where(temp<out_lower).count()
            issues = ''
            if max_val>out_upper : issues+='outlier_h, '
            if min_val<out_lower : issues+='outlier_l, '
            if missing_value > 0 : issues+='missing'
            
            rep_num[attr] = {
                'Count' : count,
                'Missing (%)' : round(missing_value/count * 100,2),
                'Cardinality (#)' : len(distinct),

                'Min' : min_val,
                'Q1' : round(qr1,2),
                'Median' : round(qr2,2),
                'Q3' : round(qr3,2),
                'Max' : max_val,
                'Mean' : round(mean_val,2),
                'Std.Dev' : round(std_val,2),
                'Outlier_lower' : out_lower,
                'Outlier_upper' : out_upper,
                'Number of outliers' : outliers_num,
                'Outliers (%)' : round(outliers_num/count * 100,2),
                'Note' : issues
                }
        df_rep_num = pd.DataFrame.from_dict(rep_num,orient='index')
        df_rep_num.index.name='Feature'
        return df_rep_num

    def categorical_report(df:pd.DataFrame)-> pd.DataFrame:
        rep_cat = {}
        for attr in df.columns:
            temp = df[attr]
            distinct = temp.unique()
            count = temp.size
            temp_val = temp.value_counts()
            missing_value = temp.isnull().sum()
            rep_cat[attr] = {
                'Count' : count,
                'Missing (%)' : round(missing_value/count * 100,2),
                'Cardinality (#)' : len(distinct),

                'Mode' : temp_val.index[0],
                'Mode Freq' : temp_val.values[0],
                'Mode (%)' : round(temp_val.values[0]/count*100,2),
                '2nd Mode' : temp_val.index[1],
                '2nd Mode Freq' : temp_val.values[1],
                '2nd Mode (%)' : round(temp_val.values[1]/count*100,2),
                'Note' : ''
                }
        df_rep_cat = pd.DataFrame.from_dict(rep_cat,orient='index')
        df_rep_cat.index.name='Feature'
        return df_rep_cat

    def clamp_data(df:pd.DataFrame, df_report:pd.DataFrame):
        df_clamped = df.copy()
        for attr in df.columns:
            temp = df[attr]
            lower_bound = df_report['Outlier_lower'][attr]
            upper_bound = df_report['Outlier_upper'][attr]
            # print('{}, {}'.format(lower_bound, upper_bound))
            df_clamped.loc[df_clamped[attr]<lower_bound]=lower_bound
            df_clamped.loc[df_clamped[attr]>upper_bound]=upper_bound
        return df_clamped

    def observation_matrix(df:pd.DataFrame, target):
        res = {}
        for target_level  in df[target].unique():
            df_temp = df[df[target]==target_level]
            level={}
            for feature in df.columns:
                for feature_level in df[feature].unique():
                    level['{}:{}'.format(feature,feature_level)] = len(df_temp[df_temp[feature]==feature_level])
                    pass
            res[target_level] = level
        return pd.DataFrame.from_dict(res,orient='index')

    def feature_observation_matrix(df:pd.DataFrame, target):
        observations=[]
        for feature in df.columns:
            if(target != feature):
                levels=[]
                rows_labels=[]
                for feature_level in df[feature].unique():
                    rows_labels.append(feature_level)
                    targets=[]
                    columns_labels = []
                    for target_level in df[target].unique():
                        columns_labels.append(target_level)
                        if(type(feature_level)==str):
                            targets.append(len(df.query('({} == {}) and ({} == "{}")'.format(target,target_level,feature,feature_level))))
                        else:
                            targets.append(len(df.query('({} == {}) and ({} == {})'.format(target,target_level,feature,feature_level))))

                    levels.append(targets)
                observations.append({'feature_name':feature,'rows_label': rows_labels,'columns_label':columns_labels,'observation_matrix':np.array(levels)})
        return observations
    
    def _entropy_of_observation_matrix(matrix:np.array):
        matrix = reshape(matrix,(1,matrix.size))
        matrix = matrix/np.sum(matrix)
        return sum([0 if x ==0 else x * np.log2(1/x) for x in matrix[0]])

    def _entropy_of_observation_matrix_sorted(observations):
        entropies = []
        for observation in observations:
            observation['entropy'] = DataManager._entropy_of_observation_matrix(observation['observation_matrix'])
            entropies.append(observation)
        return sorted(entropies, key= lambda i:i['entropy'], reverse=True)
    
    def entropies_of_dataframe_features(df, target):
        ''' Return a sorted list of features entropy from high to low. 
            higher entropy means more relative to target
        '''
        observations = DataManager.feature_observation_matrix(df,target)
        return DataManager._entropy_of_observation_matrix_sorted(observations)

    def pairwise_similarity(df, features, target):
        level_sizes={}
        for feature in features:
            feat_size = len(df[feature].unique())
            if feat_size in level_sizes:
                level_sizes[feat_size]+=1 
            else: 
                level_sizes[feat_size]=1
        for level_size in level_sizes:
            print('key: {}, value: {}'.format(level_size, level_sizes[level_size]))
            

if __name__ == '__main__':
    df_data = pd.read_csv('./data/data.csv')
    df_info = pd.read_csv('./data/data_info.csv')
    data_report = DataManager.data_analyser(df_data, df_info['type'].to_list())
    # print(data_report)
    # df_num = DataManager.extract_numerical_data(df_data, data_report['df'])
    df_cat = DataManager.extract_categorical_data(df_data, data_report['df'])
    # DataManager.pairwise_similarity(df_cat,df_cat.columns,'damage_grade')
    entropies = DataManager.entropies_of_dataframe_features(df_cat,'damage_grade')
    print('')
    # for i in entropies:
    #     print(i)
    # df_obs_matrix.to_csv('observation.csv')

    # df_rep_num = DataManager.numerical_report(df_num, 1.5, out_method='qrt')
    # df_num_cleaned = DataManager.clamp_data(df_num, df_rep_num)
    # print(df_rep_num)
    # print(DataManager.numerical_report(df_num_cleaned, 1.5, out_method='qrt'))

