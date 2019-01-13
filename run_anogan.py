"""Test `AnoGAN` Interactively.
"""

# %% Pre-load ----------------------------------------------------------------

import tensorflow as tf
import os
import shutil
print(os.getcwd())

os.chdir(
    './git/network_anomaly_detection',
)

print(tf.__version__)
import math as math
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from glob import glob
import pandas as pd
from src.base.data_processing import (data_loader,
                                      daily_kpi_data_preprocessor,
                                      datetime_setter)


# %% Data Loading ------------------------------------------------------------

downlink_col_list = [
    'CQI',
    'RSRP',
    'RSRQ',
    'DL_PRB_USAGE_RATE',
]

uplink_col_list = [
    'SINR',
    'UE_TX_POWER',
    'PHR',
    #'UL_PRB_USAGE_RATE',
]

additional_col_list = [
    #'Bandwidth',
    'UE_CONN_TOT_CNT',
]

col_list = downlink_col_list + uplink_col_list + additional_col_list

col_range_dict = {
    'CQI': [0, 15],
    'RSRP': {
        'SS': [-145, -40],  # [-140, -40],
        'ELG': [0, 50],
        'NSN': [0, 0],
    },
    'RSRQ': {
        'SS': [-20, -3.0],  # [-20, -3.5],
        'ELG': [0, 50],
        'NSN': [0, 0],
    },
    'DL_PRB_USAGE_RATE': [0, 100],
    'SINR': {
        'SS': [-30, 40],  # [-23, 35],
        'ELG': [-7, 20],  # [-5, 17],
        'NSN': [-15, 30],  # [-10, 26],
    },  # SS: [-23, 35], ELG: [-5, 17], NSN: [-10, 26]
    'UE_TX_POWER': [-17, 23],
    'PHR': [0, 65],  # [0, 63],  # [-23, 40],
    'UE_CONN_TOT_CNT': [0, 80],  # Max in data == 42.
    'HOUR': [8, 21],
}

# %% Data Loading ------------------------------------------------------------

cell_list = pd.DataFrame(
    list(
        map(
            lambda filename: filename.split('/')[-1].split('_')[:2],
            # glob('data/TOY_X/[0-9]*.csv'),
            glob('data/TOY_X_TARGET/[0-9]*.csv'),
        ),
    ),
    columns=['enb_id', 'cell_id'],
).drop_duplicates()


file_path_list = ['data/TOY_X', 'data/1808_TOY_X']
file_path_list = ['data/TOY_X_TARGET']

y_lim = (-1., 1.)

# DEFINE FLEXIBLE MAX_RANGE for `UE_CONN_TOT_CNT` --------------------
data_list_07 = []
data_list_08 = []
for enb_id, cell_id in cell_list.values.tolist():

    # data_list_07 += [
    #     data_loader('data/TOY_X', enb_id=enb_id, cell_id=cell_id)
    # ]
    # data_list_08 += [
    #     data_loader('data/1808_TOY_X', enb_id=enb_id, cell_id=cell_id)
    # ]
    data_list_08 += [
        data_loader('data/TOY_X_TARGET', enb_id=enb_id, cell_id=cell_id)
    ]

full_data_for_range = pd.concat(data_list_07 + data_list_08)


# FILTER COLUMNS
full_data_for_range['ENB_ID'] = full_data_for_range['ENB_ID'].astype(str)
full_data_for_range['CELL_ID'] = full_data_for_range['CELL_ID'].astype(str)
#full_data_for_range['CELL_ID'] = full_data_for_range['CELL_ID'].str.zfill(2)
check_col_is_null = full_data_for_range.groupby(['ENB_ID', 'CELL_ID']).agg(lambda x: x.isnull().all())


range_table = full_data_for_range.groupby(['ENB_ID', 'CELL_ID']).agg(
    [
        np.min,
        np.max,
    ]
)
ue_conn_max = range_table['UE_CONN_TOT_CNT'][['amax']]
ue_conn_max['max_range'] = 50 * (ue_conn_max // 50 + 1)

# %%-------------------------------------------------------------------

enb_id, cell_id = cell_list.values.tolist()[0]
for enb_id, cell_id in cell_list.values.tolist():

    try:
        cell_dir = f'./model_save/anogan/{enb_id}_{cell_id.zfill(2)}'
        if not os.path.isdir(cell_dir):
            max_range = ue_conn_max.loc[(str(enb_id), str(cell_id)), 'max_range']
            col_range_dict['UE_CONN_TOT_CNT'] = [0, max_range]


            tmp_c = check_col_is_null.loc[(str(enb_id), str(cell_id)), :]
            notnull_col_list = tmp_c[tmp_c < 1].index.tolist()
            model_col_list = list(
                filter(
                    lambda item: item in notnull_col_list,
                    col_list,
                )
            )
            print(len(model_col_list), model_col_list)


            (train_x_0, test_x_0,
             train_y_0, test_y_0,
             full_scaler_dict) = daily_kpi_data_preprocessor(
                # file_path='data/TOY_X',
                file_path='data/TOY_X_TARGET',
                col_list=model_col_list,
                enb_id=enb_id,
                cell_id=cell_id,
                # start_dt='2018-07-02',
                # end_dt='2018-07-31',
                # start_hour=8,
                # end_hour=21,
                start_dt='2018-11-01',
                end_dt='2018-12-13',
                start_hour=0,
                end_hour=24,
                base_freq='10S',
                resample_freq='2min',
                col_range_dict=col_range_dict,
                #use_clip=False,
                scaling_range=y_lim,
                weekday='all',  # 'weekday', 'weekend', 'all'
                x_sequence_length=64,
                y_sequence_length=0,
                foresight=1,
                test_ratio=.00,
            )

            (train_x_1, test_x_1,
             train_y_1, test_y_1,
             full_scaler_dict) = daily_kpi_data_preprocessor(
                # file_path='data/1808_TOY_X',
                file_path='data/TOY_X_TARGET',
                col_list=model_col_list,
                enb_id=enb_id,
                cell_id=cell_id,
                # start_dt='2018-08-01',
                # end_dt='2018-08-18',
                # start_hour=8,
                # end_hour=21,
                start_dt='2018-10-01',
                end_dt='2018-10-31',
                start_hour=0,
                end_hour=23,
                base_freq='10S',
                resample_freq='2min',
                col_range_dict=col_range_dict,
                #use_clip=False,
                scaling_range=y_lim,
                weekday='all',  # 'weekday', 'weekend', 'all'
                x_sequence_length=64,
                y_sequence_length=0,
                foresight=1,
                test_ratio=.00,
            )

            train_x = np.concatenate([train_x_0, train_x_1], axis=0)

            (test_x, _,
             test_y, _,
             _,) = daily_kpi_data_preprocessor(
                file_path='data/1808_TOY_X',
                col_list=model_col_list,
                enb_id=enb_id,
                cell_id=cell_id,
                # enb_id=enb_id,
                # cell_id=cell_id,
                start_dt='2018-08-19',
                end_dt='2018-08-26',
                start_hour=8,
                end_hour=21,
                base_freq='10S',
                resample_freq='2min',
                col_range_dict=col_range_dict,
                #use_clip=False,
                scaling_range=y_lim,
                weekday='all',  # 'weekday', 'weekend', 'all'
                x_sequence_length=64,
                y_sequence_length=0,
                foresight=1,
                test_ratio=.00,
            )
            print('INPUT IS READY')
            print(train_x.shape, test_x.shape)
    # %% Convert input shape for anoGAN --------------------------------------

            batch_size = 64

            # split into batch_size
            train_batch_num = math.floor(train_x.shape[0]/batch_size)
            train_convert_idx = train_batch_num * batch_size

            test_batch_num = math.floor(test_x.shape[0]/batch_size)
            test_convert_idx = test_batch_num * batch_size
            test_end_idx = np.subtract(
                test_x.shape[0],
                batch_size,
            )
            rebok_idx = np.subtract(
                batch_size,
                np.abs(
                    test_convert_idx - test_x.shape[0]
                )
            )

            # for train DCGAN
            train_x_3d = train_x.reshape(-1, 8, 8, len(model_col_list))
            train_z = np.random.uniform(
                low=0.,
                high=1.,
                size=(len(train_x), 100),
            ).astype(np.float32)

            # for train AnoGAN
            train_x_convert = train_x_3d[:train_convert_idx, : , : , :]
            train_z_convert = train_z[:train_convert_idx, :]

            # for evaluate AnoGAN
            test_x_3d = test_x.reshape(-1, 8, 8, len(model_col_list))

            test_x_split = test_x_3d[:test_convert_idx, : , : , :]
            test_x_end = test_x_3d[test_end_idx:, : , : , :]
            test_x_convert = np.append(
                test_x_split,
                test_x_end,
                axis=0,
            )
            test_z = np.random.uniform(
                low=0.,
                high=1.,
                size=(batch_size, 100),
            ).astype(np.float32)



            for batch_num in range(test_batch_num + 1):

                test_x_divided = test_x_convert[batch_size*batch_num : batch_size*(batch_num+1) , :, :, :] # 64의 크기로 쪼개기
                test_x_divided_dim = np.expand_dims(
                    test_x_divided,
                    axis=0,
                ) #stack 5차원

                test_x_divided.shape
                test_x_divided_dim.shape
                # test_x_list =[]
                if batch_num == 0:
                    test_x_list = test_x_divided_dim

                else:
                    test_x_list = np.concatenate((test_x_list, test_x_divided_dim))
                    test_x_list.shape

    # %% AnoGAN: Build --------------------------------------------------------

            import importlib
            # from workspaces.src.anogan import anogan_sohee_origin as ANOGAN
            from src.anogan import anogan as ANOGAN
            importlib.reload(ANOGAN)
            AnoGAN = ANOGAN.AnoGAN

            tf.reset_default_graph()
            anogan = AnoGAN(
                input_x_dtype=tf.float32,
                input_z_dtype=tf.float32,
                input_x_shape=(None, 8, 8, len(model_col_list)),
                input_z_shape=(None, 100),
                use_gpu=True,
                g_filter_dim=32,
                d_filter_dim=32,
                batch_size=64,
                buffer_size=1000,
                learning_rate=0.0002,
                adam_beta1=.5,
                validation_ratio=.0,
                ano_lambda_=.1,
                use_featuremap=True,
            )


    # %% AnoGAN: Train DCGAN --------------------------------------------------

            anogan.train_DCGAN(
                input_x=train_x_3d,
                input_z=train_z,
                epoch_num=2,
                gen_train_advantage_ratio=.0,
                gen_train_n_times=4,
                adam_beta1=.5,
                adam_beta2=.99,
                batch_size=64,
                learning_rate=.0002,
                decay_lr_step=1000,
                model_save_dir=cell_dir + '/dcgan',
                #pre_trained_path='./model_save_dcgan_origin',
                pre_trained_path=None,
                verbose=True,
            )

    # %% AnoGAN: Train AnoGAN -------------------------------------------------

            anogan.train_AnoGAN(
                input_x=train_x_convert,
                input_z=train_z_convert,
                epoch_num=2,
                batch_size=64,
                validation_ratio=.0,
                learning_rate=.0005,
                decay_lr_step=1000,
                adam_beta1=.5,
                adam_beta2=.99,
                ano_lambda=.1,
                dcgan_model_save_dir=cell_dir + '/dcgan',
                model_save_dir=cell_dir + '/anogan',
                #pre_trained_path='./share_test/1_cell_anoGAN',
                # pre_trained_path=None,
                verbose=True,
            )

   # %% Evaluate ANOGAN loop (batch_size) ------------------------------------

            for j in range(test_x_list.shape[0]):

                (loss_anomaly,
                 loss_residual,
                 loss_discrimination,
                 loss_residual_selected,
                 loss_discrimination_selected,
                 ano_G,
                 most_similar_ano_G) = anogan.evaluate_AnoGAN(

                    input_x=test_x_list[j],
                    input_z=test_z,
                    pre_trained_path_dcgan=cell_dir + '/dcgan',
                    pre_trained_path_anogan=cell_dir + '/anogan',
                    #target_epoch=30,
                )


                if j == test_x_list.shape[0]-1:
                    loss_residual_stacked = np.concatenate(
                        (loss_residual_stacked,loss_residual[rebok_idx:,:,:]),
                        axis=0,
                    )
                    loss_discrimination_stacked=np.concatenate(
                        (loss_discrimination_stacked,loss_discrimination[rebok_idx:,:,:]),
                        axis=0,
                    )
                    selected_residual_stacked=np.concatenate(
                        (selected_residual_stacked,loss_residual_selected[rebok_idx:,:,:]),
                        axis=0,
                    )
                    selected_dis_stacked=np.concatenate(
                        (selected_dis_stacked,loss_discrimination_selected[rebok_idx:,:,:]),
                        axis=0,
                    )
                    ano_G_stacked=np.concatenate(
                        (ano_G_stacked,ano_G[rebok_idx:,:,:]),
                        axis=0,
                    )
                    similar_ano_G_stacked=np.concatenate(
                        (similar_ano_G_stacked,most_similar_ano_G[rebok_idx:,:,:]),
                        axis=0,
                    )

                else:
                    if j == 0:
                        loss_residual_stacked=loss_residual
                        loss_discrimination_stacked=loss_discrimination
                        selected_residual_stacked=loss_residual_selected
                        selected_dis_stacked=loss_discrimination_selected
                        ano_G_stacked=ano_G
                        similar_ano_G_stacked=most_similar_ano_G

                    else:
                        loss_residual_stacked = np.append(
                            loss_residual_stacked,
                            loss_residual,
                            axis=0,
                        )
                        loss_discrimination_stacked = np.append(
                            loss_discrimination_stacked,
                            loss_discrimination,
                            axis=0,
                        )
                        selected_residual_stacked = np.append(
                            selected_residual_stacked,
                            loss_residual_selected,
                            axis=0,
                        )
                        selected_dis_stacked = np.append(
                            selected_dis_stacked,
                            loss_discrimination_selected,
                            axis=0,
                        )
                        ano_G_stacked = np.append(
                            ano_G_stacked,
                            ano_G,
                            axis=0,
                        )
                        similar_ano_G_stacked = np.append(
                            similar_ano_G_stacked,
                            most_similar_ano_G,
                            axis=0,
                        )

                loss_residual_convert = loss_residual_stacked.reshape([-1, len(model_col_list)])
                loss_discrimination_convert = loss_discrimination_stacked.reshape([-1, loss_discrimination_stacked.shape[-1]])
                selected_residual_convert = selected_residual_stacked.reshape([-1, len(model_col_list)])
                selected_dis_convert = selected_dis_stacked.reshape([-1, selected_dis_stacked.shape[-1]])
                ano_G_convert = ano_G_stacked.reshape([-1, len(model_col_list)])
                similar_ano_G_convert = similar_ano_G_stacked.reshape([-1, len(model_col_list)])

                np.savetxt(
                    cell_dir+'/loss_residual.csv',
                    loss_residual_convert,
                    fmt='%.18f',
                    delimiter=',',
                )
                np.savetxt(
                    cell_dir+'/loss_discrimination.csv',
                    loss_discrimination_convert,
                    fmt='%.18f',
                    delimiter=',',
                )
                np.savetxt(
                    cell_dir+'/selected_residual.csv',
                    selected_residual_convert,
                    fmt='%.18f',
                    delimiter=',',
                )
                np.savetxt(
                    cell_dir+'/selected_dis.csv',
                    selected_dis_convert,
                    fmt='%.18f',
                    delimiter=',',
                )
                np.savetxt(
                    cell_dir+'/ano_G.csv',
                    ano_G_convert,
                    fmt='%.18f',
                    delimiter=',',
                )
                np.savetxt(
                    cell_dir+'/similar_ano_G.csv',
                    similar_ano_G_convert,
                    fmt='%.18f',
                    delimiter=',',
                )

        # %% AnoGAN: summary -------------------------------------

            def scoring(
                input
                ):
                res = round(np.abs(input)/2*100,3)

                return res


            def summarize_table(
                input_,
                pred,
                loss_res,
                loss_dis,
                ano_lambda,
                col_list,
                ):

                result_frame_list = []
                for in_idx in range(input_.shape[0]):

                    input_idx = input_[in_idx, :, :, :] #test_x_3d[0]
                    pred_idx = pred[in_idx, :, :, :] #most_similar_ano_G[0]
                    loss_res_idx = loss_res[in_idx, :, :, :]
                    loss_dis_idx = loss_res[in_idx, :, :, :]
                    #print('test_idx', in_idx )

                    for kpi_idx in range(len(col_list)):

                        err = np.mean(input_idx[:,:,kpi_idx] - pred_idx[:,:,kpi_idx])
                        loss_res_kpi = np.mean(loss_res_idx[:,:,kpi_idx])
                        loss_dis_kpi = np.mean(loss_dis_idx[:,:,kpi_idx])
                        loss_anomaly_kpi = scoring(loss_res_kpi*(1-ano_lambda) + loss_dis_kpi*ano_lambda)

                        if kpi_idx == 0:
                            loss_ano_kpi_list=loss_anomaly_kpi,
                        else:
                            loss_ano_kpi_list = np.append(loss_ano_kpi_list, loss_anomaly_kpi)

                    loss_res_all = np.mean(loss_ano_kpi_list)
                    max_idx = np.argmax(loss_ano_kpi_list)
                    max = loss_ano_kpi_list[max_idx]

                    res_table = pd.DataFrame(
                        [
                            [in_idx, kpi_name, loss_ano_kpi]
                             for kpi_name, loss_ano_kpi in zip(col_list, loss_ano_kpi_list)
                        ],
                        columns=['CASE_INDEX', 'KPI', 'SCORE'])
                    result_frame_list += [res_table]

                return pd.concat(result_frame_list, ignore_index=True)

            # %% plot -------------------------------------
            def create_test_x_datetime(
                start_dt='2018-08-19',
                end_dt='2018-08-26',
                freq='1min',
                start_hour=8,
                end_hour=21,
                x_seq_length=64,
                foresight=1,
                y_seq_length=0,
                ):

                    test_x_datetime_base = pd.date_range(
                        start=start_dt,
                        end=end_dt,
                        freq=freq)
                    test_x_datetime_list = []
                    for day in test_x_datetime_base.day.unique():
                        daily_test_x_datetime = test_x_datetime_base[
                            (test_x_datetime_base.day == day)
                        ]
                        selected_test_x_datetime = daily_test_x_datetime[
                            (int(start_hour) <= daily_test_x_datetime.hour) &
                            (daily_test_x_datetime.hour < int(end_hour))
                        ]
                        selected_test_x_datetime = selected_test_x_datetime.to_frame()[x_seq_length + foresight + y_seq_length:]

                        test_x_datetime_list += [selected_test_x_datetime]
                        test_x_datetime = pd.concat(test_x_datetime_list)

                        test_x_datetime.columns = ['TIME_INDEX']
                        test_x_datetime['CASE_INDEX'] = range(test_x_datetime.shape[0])
                        test_x_datetime.reset_index(inplace=True)

                    return test_x_datetime

            test_x_datetime =  create_test_x_datetime(
                start_dt='2018-08-19',
                end_dt='2018-08-27',
                freq='2min',
                start_hour=8,
                end_hour=21,
                x_seq_length=64,
                foresight=1,
                y_seq_length=0,
            )
            test_x_datetime.shape

            result_table = summarize_table(test_x_3d,#real_test_convert,
                similar_ano_G_stacked,
                selected_residual_stacked,
                selected_dis_stacked,
                0.1,
                model_col_list,
            )

            result_table
            result_table = result_table.merge(
                test_x_datetime[['CASE_INDEX', 'TIME_INDEX']],
                on=['CASE_INDEX'],
                how='left',
            )[['TIME_INDEX', 'KPI', 'SCORE']]

            result_table.set_index('TIME_INDEX', inplace=True)
            result_table

            print('save_summary_table', cell_dir)


            summ_table = result_table.pivot_table(
                index=result_table.index,
                columns=['KPI'],
                aggfunc=np.mean,
                margins=False,
            )# ['SCORE']
            summ_table[('REPR', 'WORST_KPI')] = summ_table['SCORE'].idxmax(
                axis=1
            )
            summ_table[('REPR', 'WORST_KPI_SCORE')] = summ_table['SCORE'].max(
                axis=1
            )
            summ_table[('REPR', 'TOTAL')] = summ_table['SCORE'].mean(
                axis=1
            )
            # summ_table.to_csv(cell_dir + '/summary_table.csv')


    except KeyError:
        continue

# %% load ascore table --------------------------------------------------------

data_list = []

for enb_id, cell_id in cell_list.values.tolist():

    data = data_loader('data/1808_TOY_Y', enb_id=enb_id, cell_id=cell_id)
    data[['ENB_ID', 'CELL_ID', 'VEND_ID']] = data[['ENB_ID', 'CELL_ID', 'VEND_ID']].astype(str)
    data['CELL_ID'] = data['CELL_ID'].str.zfill(2)
    selected = datetime_setter(
        data,
        start_dt='2018-08-19',
        end_dt='2018-08-26',
        start_hour=8,
        end_hour=21,
        freq='2min',
    )
    data_list += [selected]

concatted = pd.concat(data_list)

concatted.head()
# %%

ascore_table = concatted.groupby(['ENB_ID', 'CELL_ID'])[['ASCORE',
                                                        'CQI',
                                                        'RSRP',
                                                        'RSRQ',
                                                        'DL_PRB_USAGE_RATE',
                                                        'PHR','SINR',
                                                        'UE_TX_POWER']].resample('2min').pad()


ascore_table.index.names = ['ENB_ID', 'CELL_ID', 'TIME_INDEX']
ascore_table.columns = pd.MultiIndex.from_tuples([('COMPARE', 'ASCORE'),
                                                    ('COMPARE', 'CQI'),
                                                    ('COMPARE', 'RSRP'),
                                                    ('COMPARE', 'RSRQ'),
                                                    ('COMPARE', 'DL_PRB_USAGE_RATE'),
                                                    ('COMPARE', 'PHR'),
                                                    ('COMPARE', 'SINR'),
                                                    ('COMPARE', 'UE_TX_POWER'),
                                                    ])

ascore_table
# %% merge ascore_table&summary_table ----------------------------------------

summary_file_list = glob('model_save/anogan/*/summary_table.csv')

summary_table_list = []
for summary_filename in summary_file_list:
    enb_id, cell_id = summary_filename.split('/')[2].split('_')
    summary = pd.read_csv(
        summary_filename,
        index_col=[0],
        header=[0, 1],
        parse_dates=True,
        mangle_dupe_cols=True,
    )
    summary['ENB_ID'] = str(enb_id)
    summary['CELL_ID'] = str(cell_id)
    #data['VEND_ID'] = 'SS'
    summary.set_index(
        ['ENB_ID', 'CELL_ID'],
        append=True,
        inplace=True,
        drop=True,
    )
    summary_table_list += [summary]
summary_table_list

summary_table = pd.concat(summary_table_list)
summary_table.head()

final_summary_table = pd.merge(
    summary_table.reset_index(),
    ascore_table.reset_index(),
    how='left',
    on=['TIME_INDEX', 'ENB_ID', 'CELL_ID'],

).set_index(
    ['ENB_ID', 'CELL_ID', 'TIME_INDEX'],
    drop=True,
).sort_index()
final_summary_table

# final_summary_table.to_csv('model_save/anogan/final_summary.csv')
