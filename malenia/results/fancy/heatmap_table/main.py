import pandas as pd

from MCM import MCM


if __name__ == "__main__":

    path_res = '/home/rayllon/INCEPTION/orchestator/tsoc_I_results.csv'
    output_dir = './'

    df_results = pd.read_csv(path_res)

    MCM.compare(
        df_results=df_results,
        order_WinTieLoss="higher",
        used_statistic="QWK",
        fig_savename='heatmap',
        load_analysis=False,
        order_better="decreasing",
        colorbar_orientation='vertical',
        precision=4,
        excluded_col_comparates=['XGBoost','LogReg','TSF'],
        excluded_row_comparates=['XGBoost','LogReg','TSF'],
    )

    # MCM.compare(
    #     df_results=df_results,
    #     excluded_col_comparates=['clf1','clf3'],
    #     fig_savename='heatline_vertical',
    #     load_analysis=False
    # )

    # MCM.compare(
    #     df_results=df_results,
    #     excluded_row_comparates=['clf1','clf3'],
    #     fig_savename='heatline_horizontal',
    #     load_analysis=False
    # )