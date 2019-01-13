"""Get sliding windows from the real data. (Revised: 2018-09-03 00:30:01)

"""

import os
import shutil
from glob import glob

# !python download.py celebA

import numpy as np
import pandas as pd


# %% Data Loading ------------------------------------------------------------

def data_loader(
    base_path,
    enb_id=None,
    cell_id=None,
    #col_list=None,
    ):

    #print('Columns: %s' % col_list)

    # Get a file list.
    file_list = sorted(
        glob(
            os.path.join(
                base_path,
                "{ENB_ID}_{CELL_ID}_*.csv".format(
                    ENB_ID=enb_id,
                    CELL_ID=cell_id,
                ),
            ),
        ),
    )
    print('Found %s Files.' % len(file_list))

    # Load it.
    full_data = pd.concat(map(pd.read_csv, file_list))
    data_tmidx = pd.to_datetime(
        full_data['EVT_DTM'],
        format='%Y%m%d%H%M%S'
    )
    full_data.set_index(data_tmidx, inplace=True)

    # Drop the Duplicated.
    full_data.drop_duplicates(
        subset='EVT_DTM',
        keep='last',
        inplace=True,
    )

    return full_data


#
# cell_list = [
#     ('28339', '11'),
#     ('28355', '21'),
#     ('28063', '12'),
# ]
#
# data = pd.concat(
#     [data_loader(
#         'data/TOY_X', enb_id=enb, cell_id=cell,
#     ) for enb, cell in cell_list]
# )
# data.shape


# %% DatetimeIndex Setting ---------------------------------------------------


def datetime_setter(
    data,
    start_dt=None,
    end_dt=None,
    start_hour=None,
    end_hour=None,
    freq='10S',
    ):

    full_datetime = pd.date_range(
        start=str(start_dt),
        end=str(end_dt),
        freq=freq,
    )

    selected_datetime = full_datetime[
        (int(start_hour) <= full_datetime.hour) &
        (full_datetime.hour < int(end_hour))
    ]


    tmp_table = pd.DataFrame([],
        index=selected_datetime,
        #columns=col_list,
    )


    joined = tmp_table.join(
        data,
        how='left',
        sort=True,
    )

    if 'UE_CONN_TOT_CNT' in joined.columns:
        joined['UE_CONN_TOT_CNT'] = joined['UE_CONN_TOT_CNT'].astype(
            np.float32
        )

    return joined

# selected = datetime_setter(
#     data,
#     start_dt='2018-07-02',
#     end_dt='2018-07-31',
#     start_hour=8,
#     end_hour=21,
#     freq='10S',
# )
#selected = joined[joined.index.day != 13]


# %% Resampling (1 Min) ------------------------------------------------------

# resampled = selected.groupby(
#     pd.Grouper(freq='D')
# )[col_list].resample('2min').mean()
# resampled.reset_index(level=0, drop=True, inplace=True)

# %% Missing Value Handling --------------------------------------------------

# full_median = resampled[col_list].mean()
# daily_interped = resampled.groupby(pd.Grouper(freq='D'))[col_list].apply(
#     lambda grp: grp.interpolate(
#         method='nearest',
#         axis=0,
#         limit=None,
#         limit_direction='both',
#     )
# )
#
# daily_filled = resampled.groupby(pd.Grouper(freq='D'))[col_list].apply(
#     lambda grp: grp.ffill().bfill()
# )


# %% Column Scaling: Range ---------------------------------------------------

class RangeScaler(object):

    def __init__(
        self,
        real_range=(-10., 10.),
        feature_range=(0., 1.),
        copy=True,
        ):

        self.base_range = np.array((0., 1.))
        self.real_range = np.array(real_range, dtype=np.float32)
        self.feature_range = np.array(feature_range, dtype=np.float32)
        self.copy = copy

        self.real_min, self.real_max = real_range
        self.real_length = real_range[1] - real_range[0]

        self.feature_min, self.feature_max = feature_range
        self.feature_length = feature_range[1] - feature_range[0]
        self.feature_mean = np.mean(feature_range)


    def _check_array(self, input_x):
        if isinstance(input_x, np.ndarray):
            return input_x
        elif isinstance(input_x, (pd.Series, pd.DataFrame)):
            return input_x.values
        else:
            raise TypeError(
            "Input must be one of this: " +
            "['numpy.ndarray', 'pandas.Series', 'pandas.DataFrame']"
            )


    def _partial_scale(self, input_x):

        input_x = self._check_array(input_x)
        base_ranged = (
            (input_x - self.real_min) / self.real_length
        )

        if np.array_equal(self.feature_range, self.base_range):
            return base_ranged
        else:
            return (base_ranged * self.feature_length) + self.feature_min


    def transform(self, input_x):
        return self._partial_scale(input_x)


    def inverse_transform(self, scaled_x):
        base_ranged = (scaled_x - self.feature_min) / self.feature_length

        if np.array_equal(self.feature_range, self.base_range):
            return base_ranged
        else:
            return (base_ranged * self.real_length) + self.real_min


col_range_dict = {
    'CQI': [0, 15],
    'RSRP': [-140, -40],
    'RSRQ': [-20, -3.5],
    'DL_PRB_USAGE_RATE': [0, 100],
    'SINR': {
        'SS': [-23, 35],
        'ELG': [-5, 17],
        'NSN': [-10, 26],
    },  # SS: [-23, 35], ELG: [-5, 17], NSN: [-10, 26]
    'UE_TX_POWER': [-17, 23],
    'PHR': [0, 63],  # [-23, 40],
    'UE_CONN_TOT_CNT': [0, 80],
    'HOUR': [8, 21],
}


def column_range_scaler(
    dataframe,
    vendor_name,
    col_real_range_dict=None,
    feature_range=(-1., 1.),
    use_clip=False,
    ):

    if set(dataframe.columns) - set(col_range_dict):
        raise AttributeError(
            "'col_real_range_dict' should contains all columns in 'dataframe'"
        )
    else:
        result_frame = dataframe.copy()
        scaler_dict = dict()
        #for col, real_range in col_real_range_dict.items():
        for col in dataframe.columns:

            if col in ('SINR', 'RSRP', 'RSRQ'):
                col_real_range = col_real_range_dict[col][vendor_name]
            else:
                col_real_range = col_real_range_dict[col]

            scaler = RangeScaler(
                real_range=col_real_range,
                feature_range=feature_range,
            )
            # if result_frame[col].isnull().all():
            #     result_frame[col].fillna(
            #         np.mean(col_real_range),
            #         inplace=True,
            #     )
            scaled = scaler.transform(result_frame[[col]])

            scaler_dict[col] = scaler

            if result_frame[col].isnull().all():
                result_frame[col] = np.mean(feature_range)
            else:
                result_frame[col] = scaled

        return result_frame, scaler_dict


# full_scaled, full_scaler_dict = column_range_scaler(
#     daily_filled,
#     col_real_range_dict=col_range_dict,
#     feature_range=(-0.7, 1.),
# )

# %% Choose Only One Weekday for Test ----------------------------------------

# 0 for Monday, 6 for Sunday
#
# weekday_data = full_scaled[
#     full_scaled.index.weekday <= 4
# ]
# weekend_data = full_scaled[
#     full_scaled.index.weekday > 4
# ]
# #targetday_data = weekend_data
# #targetday_data = weekday_data
# targetday_data = full_scaled
#
# targetday_data['CQI'].describe()

# %% Window Sliding ----------------------------------------------------------

def daily_sliding_window_extractor(
    dataframe,
    x_col_list=None,
    y_col_list=None,
    x_len=5,
    y_len=3,
    foresight=1,
    test_ratio=.3,
    return_timeindex=False,
    ):

    ndim_x = len(x_col_list)
    ndim_y = len(y_col_list)

    full_train_x_arr = np.empty((0, x_len, ndim_x), dtype=np.float32)
    full_test_x_arr = np.empty((0, x_len, ndim_x), dtype=np.float32)
    full_train_y_arr = np.empty((0, y_len, ndim_y), dtype=np.float32)
    full_test_y_arr = np.empty((0, y_len, ndim_y), dtype=np.float32)

    full_train_x_timeindex = np.empty((0, ), dtype=np.float32)
    full_test_x_timeindex = np.empty((0, ), dtype=np.float32)
    full_train_y_timeindex = np.empty((0, ), dtype=np.float32)
    full_test_y_timeindex = np.empty((0,  ), dtype=np.float32)
    # full_train_x_list = []
    # full_test_x_list = []
    # full_train_y_list = []
    # full_test_y_list = []

    grouped = dataframe.groupby(pd.Grouper(freq='D'))
    for day, group in grouped:

        minimum_size = x_len + y_len + foresight

        # Split the Data: Train & Test
        train_ratio = 1 - test_ratio
        train_size = int(group.shape[0] * train_ratio)

        if train_size < minimum_size:
            continue

        else:
            source_x = group[x_col_list].sort_index().values.reshape(
                -1, ndim_x
            ).astype('float32')
            source_y = group[y_col_list].sort_index().values.reshape(
                -1, ndim_y
            ).astype('float32')

            # Reshape the Data: [samples, time steps, features]
            """
            >>> xlen, foresight = 3, 2
            >>> tmp
            array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

            >>> x = np.array([tmp[i:i+3] for i in range(len(tmp) - 3 - 2)])
            >>> x
            array([[0, 1, 2],
                   [1, 2, 3],
                   [2, 3, 4],
                   [3, 4, 5],
                   [4, 5, 6],
                   [5, 6, 7],
                   [6, 7, 8]])

            """
            slided_x = np.array(
                [
                    source_x[i:i+x_len]
                    for i in range(
                        0,
                        len(source_x) - x_len - foresight - y_len,
                    )
                ]
            )
            # slided_x_timeindex = group[x_col_list].sort_index().index[
            #     range(0, len(source_x) - x_len - foresight - y_len,)
            # ]

            """
            >>> y = tmp[3 + 2:]
            >>> y
            array([ 5,  6,  7,  8,  9, 10, 11])

            """
            # slided_y = source_y[x_len + foresight:-y_len]
            y_start_idx = x_len + foresight
            slided_y = np.array(
                [
                    source_y[i:i+y_len]
                    for i in range(
                        y_start_idx,
                        len(source_y) - y_len,
                    )
                ]
            )
            # slided_y_timeindex = group[y_col_list].sort_index().index[
            #     range(y_start_idx, len(source_y) - y_len,)
            # ]

            train_x, test_x = slided_x[:train_size], slided_x[train_size:]
            train_y, test_y = slided_y[:train_size], slided_y[train_size:]

            # train_x_timeindex = slided_x_timeindex[:train_size]
            # test_x_timeindex = slided_x_timeindex[train_size:]
            # train_y_timeindex = slided_y_timeindex[:train_size]
            # test_y_timeindex = slided_y_timeindex[train_size:]

            # full_train_x_list.append(train_x)
            # full_test_x_list.append(test_x)
            # full_train_y_list.append(train_y)
            # full_test_y_list.append(test_y)
            full_train_x_arr = np.concatenate(
                [
                    full_train_x_arr,
                    train_x,
                ],
                axis=0,
            )
            full_test_x_arr = np.concatenate(
                [
                    full_test_x_arr,
                    test_x,
                ],
                axis=0,
            )
            full_train_y_arr = np.concatenate(
                [
                    full_train_y_arr,
                    train_y,
                ],
                axis=0,
            )
            full_test_y_arr = np.concatenate(
                [
                    full_test_y_arr,
                    test_y,
                ],
                axis=0,
            )

            # # TimeIndex
            # full_train_x_timeindex = np.concatenate(
            #     [
            #         full_train_x_timeindex,
            #         train_x,
            #     ],
            #     axis=0,
            # )
            # full_test_x_timeindex = np.concatenate(
            #     [
            #         full_test_x_arr,
            #         test_x,
            #     ],
            #     axis=0,
            # )
            # full_train_x_timeindex = np.concatenate(
            #     [
            #         full_train_y_arr,
            #         train_y,
            #     ],
            #     axis=0,
            # )
            # full_test_y_timeindex = np.concatenate(
            #     [
            #         full_test_y_timeindex,
            #         test_y,
            #     ],
            #     axis=0,
            # )

    if return_timeindex:
        res = (
            full_train_x_arr,
            full_test_x_arr,
            full_train_y_arr,
            full_test_y_arr,
            # full_train_x_timeindex,
            # full_test_x_timeindex,
            # train_y_timeindex,
            # test_y_timeindex,
        )

    else:
        res = (
            full_train_x_arr,
            full_test_x_arr,
            full_train_y_arr,
            full_test_y_arr,
        )

    print('X :', full_train_x_arr.shape, full_test_x_arr.shape)
    print('Y :', full_train_y_arr.shape, full_test_y_arr.shape)

    return res


# x_sequence_length = 10
# y_sequence_length = 5
#
# (train_x, test_x,
#  train_y, test_y) = daily_sliding_window_extractor(
#     targetday_data,
#     x_col_list=col_list, #[:1],
#     y_col_list=col_list, #[:1],
#     x_len=x_sequence_length,
#     y_len=y_sequence_length,
#     foresight=1,
#     test_ratio=.2,
# )


# %% A Data Preprocessor -----------------------------------------------------

def daily_kpi_data_preprocessor(
    file_path='data/TOY_X',
    col_list=None,
    enb_id='28339',
    cell_id='11',
    start_dt='2018-07-02',
    end_dt='2018-07-31',
    start_hour=8,
    end_hour=21,
    base_freq='10S',
    resample_freq='2min',
    col_range_dict=None,
    scaling_range=(-0.7, 1.),
    use_clip=False,
    weekday='all',  # 'weekday', 'weekend', 'all'
    x_sequence_length=10,
    y_sequence_length=5,
    foresight=1,
    test_ratio=.2,
    return_timeindex=False,
    ):

    # Data Loading
    data = data_loader(
        file_path,
        enb_id=enb_id,
        cell_id=cell_id,
    )

    # DatetimeIndex Setting
    selected = datetime_setter(
        data,
        start_dt=start_dt,
        end_dt=end_dt,
        start_hour=start_hour,
        end_hour=end_hour,
        freq=base_freq,
    )

    # VENDOR name
    vendor_name = selected['VEND_ID'].unique()[0]

    # Resampling (1 Min)
    resampled = selected.groupby(
        pd.Grouper(freq='D')
    )[col_list].resample(resample_freq).mean()
    resampled.reset_index(level=0, drop=True, inplace=True)

    # Missing Value Handling
    # daily_interped = resampled.groupby(pd.Grouper(freq='D'))[col_list].apply(
    #     lambda grp: grp.interpolate(
    #         method='nearest',
    #         axis=0,
    #         limit=None,
    #         limit_direction='both',
    #     )
    # )
    daily_filled = resampled.groupby(pd.Grouper(freq='D'))[col_list].apply(
        lambda grp: grp.ffill().bfill()
    )
    # col_median = daily_filled[col_list].median(axis=0, skipna=True)
    daily_filled = daily_filled.ffill().bfill()
    daily_filled = daily_filled.fillna(0.)

    # Column Scaling: Range
    full_scaled, full_scaler_dict = column_range_scaler(
        daily_filled,
        vendor_name,
        col_real_range_dict=col_range_dict,
        feature_range=scaling_range,
    )

    # Choose Only One Weekday for Test
    if weekday == 'weekday':
        targetday_data = full_scaled[
            full_scaled.index.weekday <= 4
        ]
    elif weekday == 'weekend':
        targetday_data = full_scaled[
            full_scaled.index.weekday > 4
        ]
    else:
        targetday_data = full_scaled

    # Window Sliding
    if return_timeindex:
        (train_x, test_x,
         train_y, test_y,
         train_x_timeindex,
         test_x_timeindex,
         train_y_timeindex,
         test_y_timeindex) = daily_sliding_window_extractor(
            targetday_data,
            x_col_list=col_list, #[:1],
            y_col_list=col_list, #[:1],
            x_len=x_sequence_length,
            y_len=y_sequence_length,
            foresight=foresight,
            test_ratio=test_ratio,
            return_timeindex=return_timeindex,
        )
        return (
            train_x, test_x, train_y, test_y, full_scaler_dict,
            train_x_timeindex,
            test_x_timeindex,
            train_y_timeindex,
            test_y_timeindex,
        )

    else:
        (train_x, test_x,
         train_y, test_y,) = daily_sliding_window_extractor(
            targetday_data,
            x_col_list=col_list, #[:1],
            y_col_list=col_list, #[:1],
            x_len=x_sequence_length,
            y_len=y_sequence_length,
            foresight=foresight,
            test_ratio=test_ratio,
            return_timeindex=return_timeindex,
        )
        return train_x, test_x, train_y, test_y, full_scaler_dict


# %% Usage -------------------------------------------------------------------

if __name__ == '__main__':

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

    file_path = 'data/TOY_X'
    col_list = downlink_col_list + uplink_col_list + additional_col_list
    enb_id = '28339'
    cell_id = '11'
    start_dt='2018-07-02',
    end_dt='2018-07-31',
    start_hour=8,
    end_hour=21,
    base_freq='10S'
    resample_freq='2min'

    col_range_dict = {
        'CQI': [0, 15],
        'RSRP': [-140, -40],
        'RSRQ': [-20, -3.5],
        'DL_PRB_USAGE_RATE': [0, 100],
        'SINR': [-23, 35],  # SS: [-23, 35], ELG: [-5, 17], NSN: [-10, 26]
        'UE_TX_POWER': [-17, 23],
        'PHR': [0, 63],  # [-23, 40],
        'UE_CONN_TOT_CNT': [0, 60],  # Max in data == 42.
        'HOUR': [8, 21],
    }
    scaling_range = (-0.7, 1.)
    weekday = 'all'  # 'weekday', 'weekend', 'all'

    x_sequence_length = 10
    y_sequence_length = 5
    foresight=1,
    test_ratio=.2,

    # (train_x, test_x,
    #  train_y, test_y,
    #  full_scaler_dict) = daily_kpi_data_preprocessor(
    #     col_list=col_list,
    #     enb_id='28339',
    #     cell_id='11',
    #     start_dt='2018-07-02',
    #     end_dt='2018-07-31',
    #     start_hour=8,
    #     end_hour=21,
    #     base_freq='10S',
    #     resample_freq='2min',
    #     col_range_dict=col_range_dict,
    #     use_clip=False,
    #     scaling_range=(-0.7, 1.),
    #     weekday='all',  # 'weekday', 'weekend', 'all'
    #     x_sequence_length=10,
    #     y_sequence_length=5,
    #     foresight=1,
    #     test_ratio=.2,
    # )

    (train_x, test_x,
     train_y, test_y,
     full_scaler_dict) = daily_kpi_data_preprocessor(
        col_list=col_list,
        enb_id='28339',
        cell_id='11',
        start_dt='2018-07-02',
        end_dt='2018-07-31',
        start_hour=8,
        end_hour=21,
        base_freq='10S',
        resample_freq='2min',
        col_range_dict=col_range_dict,
        use_clip=False,
        scaling_range=(-0.7, 1.),
        weekday='all',  # 'weekday', 'weekend', 'all'
        x_sequence_length=10,
        y_sequence_length=5,
        foresight=1,
        test_ratio=.2,
    )
