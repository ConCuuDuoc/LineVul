import h5py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

file_path = '../data/raw-draper/VDISC_train.hdf5'
processed_file_path = 'processed_VDISC.hdf5'
train_file_path = 'VDISC_train.hdf5'
test_file_path = 'VDISC_test.hdf5'
val_file_path = 'VDISC_validate.hdf5'

with h5py.File(file_path, 'r') as file:
    with pd.HDFStore(processed_file_path) as store:
        dataset_keys = list(file.keys())[:-1]

        for dataset_key in dataset_keys:
            dataset = file[dataset_key]
            df = pd.DataFrame(dataset[:])

            df.dropna(inplace=True)
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

            print(f"Processed and scaled DataFrame for {dataset_key}:")
            print(df_scaled.head())

            store.put(dataset_key, df_scaled, data_columns=True)
            
            train_df, test_val_df = train_test_split(df_scaled, test_size=0.2, random_state=42)
            test_df, val_df = train_test_split(test_val_df, test_size=0.5, random_state=42)
            with pd.HDFStore(train_file_path) as train_store:
                train_store.put(dataset_key, train_df, data_columns=True)

            with pd.HDFStore(test_file_path) as test_store:
                test_store.put(dataset_key, test_df, data_columns=True)

            with pd.HDFStore(val_file_path) as val_store:
                val_store.put(dataset_key, val_df, data_columns=True)