import numpy as np
import pandas as pd
from malenia.results.recol_utils_global import means_with_stds, rankings_avgseeds
from malenia.results.recol_utils_toexcel_aux import xlsxwrite_means_stds_results, xlsxwrite_rankings_results


def create_basic_results_excel(r_results, metrics_with_gib, target_excel, geq_n_classes=6):
    excel_writer = pd.ExcelWriter(target_excel, engine="xlsxwriter")
    worksheet = excel_writer.book.add_worksheet("Means with Stds")
    metrics = list(metrics_with_gib.keys())

    xlsxwrite_means_stds_results(
        res=means_with_stds(r_results, metrics, datasets_min_n_classes=0),
        metrics_with_gib=metrics_with_gib,
        excel_writer=excel_writer,
        worksheet=worksheet,
        title="Means with Stds",
        starting_row=2,
        starting_col=2,
    )

    n_methods = len(r_results["method"].unique())

    xlsxwrite_means_stds_results(
        res=means_with_stds(r_results, metrics, datasets_min_n_classes=geq_n_classes),
        metrics_with_gib=metrics_with_gib,
        excel_writer=excel_writer,
        worksheet=worksheet,
        title=f"Means with Stds (>= {geq_n_classes} classes)",
        starting_row=n_methods + 5,
        starting_col=2,
    )

    worksheet = excel_writer.book.add_worksheet("Rankings")

    xlsxwrite_rankings_results(
        res=rankings_avgseeds(r_results, metrics_with_gib, datasets_min_n_classes=0, method="avg_seeds"),
        excel_writer=excel_writer,
        worksheet=worksheet,
        title="Rankings",
        starting_row=2,
        starting_col=2,
    )

    xlsxwrite_rankings_results(
        res=rankings_avgseeds(
            r_results, metrics_with_gib, datasets_min_n_classes=geq_n_classes, method="avg_seeds"
        ),
        excel_writer=excel_writer,
        worksheet=worksheet,
        title=f"Rankings (>= {geq_n_classes} classes)",
        starting_row=n_methods + 5,
        starting_col=2,
    )

    NUM_FORMAT = "#,##0.000"
    workbook = excel_writer.book
    base_format = workbook.add_format({"num_format": NUM_FORMAT, "align": "center", "bg_color": "#E6E6E6"})
    bold_format = workbook.add_format(
        {"bold": True, "num_format": NUM_FORMAT, "align": "center", "bottom": 1}
    )
    good_format = workbook.add_format(
        {"bg_color": "#C6EFCE", "font_color": "#006100", "num_format": NUM_FORMAT, "align": "center"}
    )
    neutral_format = workbook.add_format(
        {"bg_color": "#FFEB9C", "font_color": "#9C6500", "num_format": NUM_FORMAT, "align": "center"}
    )
    string_format = workbook.add_format({"align": "left", "right": 1})

    for metric in metrics_with_gib.keys():
        worksheet = excel_writer.book.add_worksheet(f"{metric.upper()}")
        metric_res = (
            r_results.groupby(["dataset", "method"])
            .mean()[[metric]]
            .reset_index()
            .pivot(index="dataset", columns="method", values=metric)
            .reset_index()
        )

        # Write headers
        worksheet.write(0, 0, "Dataset", bold_format)
        for col_num, method in enumerate(metric_res.columns[1:]):
            worksheet.write(0, col_num + 1, method, bold_format)

        # Check if column header contains "(std)" for italic formatting
        for row_num, row_values in enumerate(metric_res.values):

            if metrics_with_gib[metric]:
                best_val = row_values[1:].max()
                second_best_val = np.partition(row_values[1:], -2)[-2]
            else:
                best_val = row_values[1:].min()
                second_best_val = np.partition(row_values[1:], 2)[1]

            worksheet.write(row_num + 1, 0, row_values[0], string_format)  # Write dataset name

            for col_num, val in enumerate(row_values[1:]):
                if val == best_val:
                    worksheet.write(row_num + 1, col_num + 1, val, good_format)
                elif val == second_best_val:
                    worksheet.write(row_num + 1, col_num + 1, val, neutral_format)
                else:
                    worksheet.write(row_num + 1, col_num + 1, val, base_format)
        worksheet.autofit()

    excel_writer.close()


def create_spss_formatted_results(r_results, metrics_with_gib, target_excel):
    excel_writer = pd.ExcelWriter(target_excel, engine="xlsxwriter")

    NUM_FORMAT = "#,##0.000"
    workbook = excel_writer.book
    base_format = workbook.add_format({"num_format": NUM_FORMAT, "align": "center", "bg_color": "#E6E6E6"})
    bold_format = workbook.add_format(
        {"bold": True, "num_format": NUM_FORMAT, "align": "center", "bottom": 1}
    )
    string_format = workbook.add_format({"align": "left", "right": 1})

    for metric in metrics_with_gib.keys():
        worksheet = excel_writer.book.add_worksheet(f"{metric.upper()}")
        r_results["dataset_method"] = (
            r_results["dataset"].str.replace(".csv", "", regex=False) + "__" + r_results["method"]
        )
        metric_res = (
            r_results.drop(columns=["dataset", "method", "runtime"])
            .groupby(["dataset_method", "seed"])
            .mean()[[metric]]
            .reset_index()
        )
        metric_res = metric_res.pivot(index="seed", columns="dataset_method", values=metric).reset_index()

        # Write headers
        worksheet.write(0, 0, "seed", bold_format)
        for col_num, method in enumerate(metric_res.columns[1:]):
            worksheet.write(0, col_num + 1, method, bold_format)

        # Check if column header contains "(std)" for italic formatting
        for row_num, row_values in enumerate(metric_res.values):
            worksheet.write(row_num + 1, 0, row_values[0], string_format)  # Write dataset name
            for col_num, val in enumerate(row_values[1:]):
                worksheet.write(row_num + 1, col_num + 1, val, base_format)
        worksheet.autofit()

    excel_writer.close()
