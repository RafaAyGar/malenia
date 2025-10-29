from pandas.api.types import is_string_dtype, is_numeric_dtype


def xlsxwrite_means_stds_results(
    res, metrics_with_gib, excel_writer, worksheet, title="Results", starting_row=0, starting_col=0
):
    # Get the xlsxwriter workbook and worksheet objects
    workbook = excel_writer.book

    # Define formats for bold, italic, and highlighted cells
    NUM_FORMAT = "#,##0.000"
    base_format = workbook.add_format({"num_format": NUM_FORMAT, "align": "center", "bg_color": "#E6E6E6"})
    bold_format = workbook.add_format({"bold": True, "num_format": NUM_FORMAT, "align": "center", "bottom": 1})
    italic_format = workbook.add_format(
        {"italic": True, "font_size": 9, "num_format": NUM_FORMAT, "align": "center", "right": 1}
    )
    good_format = workbook.add_format(
        {"bg_color": "#C6EFCE", "font_color": "#006100", "num_format": NUM_FORMAT, "align": "center"}
    )
    neutral_format = workbook.add_format(
        {"bg_color": "#FFEB9C", "font_color": "#9C6500", "num_format": NUM_FORMAT, "align": "center"}
    )
    # create centering format

    res = res.reset_index()

    # Compute merge range
    num_cols = len(res.columns)
    merge_range = (
        starting_row - 1,
        starting_col,
        starting_row - 1,
        starting_col + num_cols - 1,
    )  # (first_row, first_col, last_row, last_col)

    # Define title format
    title_format = workbook.add_format(
        {"bold": True, "align": "center", "font_size": 12, "bg_color": "#FFC1AB", "border": 1}
    )

    # Write merged title (adjust text as needed)"
    worksheet.merge_range(*merge_range, title, title_format)

    # Apply formatting to headers (bold) and values (highlight minimum and second lowest)
    for col_num, value in enumerate(res.columns.values):
        col_values = res.iloc[:, col_num]

        if is_string_dtype(res[value]):
            worksheet.write(starting_row + 0, starting_col + col_num, value, bold_format)
            for row_num, val in enumerate(col_values):
                worksheet.write(
                    starting_row + row_num + 1,
                    starting_col + col_num,
                    val,
                    workbook.add_format({"align": "left", "right": 1}),
                )
        elif is_numeric_dtype(res[value]):
            # Apply bold formatting to column headers
            worksheet.write(starting_row + 0, starting_col + col_num, value, bold_format)

            # Highlight minimum value with "good" format, second lowest with "neutral" format
            metric_name = value.split(" (")[0]
            greater_is_better = metrics_with_gib[metric_name]
            if greater_is_better:
                best_val = col_values.max()
                second_best_val = col_values.nlargest(2).iloc[-1]
            else:
                best_val = col_values.min()
                second_best_val = col_values.nsmallest(2).iloc[-1]

            # Check if column header contains "(std)" for italic formatting
            if "(std)" in value:
                for row_num, val in enumerate(col_values):
                    worksheet.write(starting_row + row_num + 1, starting_col + col_num, val, italic_format)
            else:
                for row_num, val in enumerate(col_values):
                    if val == best_val:
                        worksheet.write(starting_row + row_num + 1, starting_col + col_num, val, good_format)
                    elif val == second_best_val:
                        worksheet.write(starting_row + row_num + 1, starting_col + col_num, val, neutral_format)
                    else:
                        worksheet.write(starting_row + row_num + 1, starting_col + col_num, val, base_format)
    worksheet.autofit()


def xlsxwrite_rankings_results(res, excel_writer, worksheet, title="Results", starting_row=0, starting_col=0):
    # Get the xlsxwriter workbook and worksheet objects
    workbook = excel_writer.book

    # Define formats for bold, italic, and highlighted cells
    NUM_FORMAT = "#,##0.000"
    base_format = workbook.add_format({"num_format": NUM_FORMAT, "align": "center", "bg_color": "#E6E6E6"})
    bold_format = workbook.add_format({"bold": True, "num_format": NUM_FORMAT, "align": "center", "bottom": 1})
    good_format = workbook.add_format(
        {"bg_color": "#C6EFCE", "font_color": "#006100", "num_format": NUM_FORMAT, "align": "center"}
    )
    neutral_format = workbook.add_format(
        {"bg_color": "#FFEB9C", "font_color": "#9C6500", "num_format": NUM_FORMAT, "align": "center"}
    )

    res = res.reset_index()

    # Compute merge range
    num_cols = len(res.columns)
    merge_range = (
        starting_row - 1,
        starting_col,
        starting_row - 1,
        starting_col + num_cols - 1,
    )  # (first_row, first_col, last_row, last_col)

    # Define title format
    title_format = workbook.add_format(
        {"bold": True, "align": "center", "font_size": 12, "bg_color": "#FFC1AB", "border": 1}
    )

    # Write merged title (adjust text as needed)"
    worksheet.merge_range(*merge_range, title, title_format)

    # Apply formatting to headers (bold) and values (highlight minimum and second lowest)
    for col_num, value in enumerate(res.columns.values):
        col_values = res.iloc[:, col_num]

        if is_string_dtype(res[value]):
            worksheet.write(starting_row + 0, starting_col + col_num, value, bold_format)
            for row_num, val in enumerate(col_values):
                worksheet.write(
                    starting_row + row_num + 1,
                    starting_col + col_num,
                    val,
                    workbook.add_format({"align": "left", "right": 1}),
                )
        elif is_numeric_dtype(res[value]):
            # Apply bold formatting to column headers
            worksheet.write(starting_row + 0, starting_col + col_num, value, bold_format)

            # Highlight minimum value with "good" format, second lowest with "neutral" format
            best_val = col_values.min()
            second_best_val = col_values.nsmallest(2).iloc[-1]

            # Check if column header contains "(std)" for italic formatting
            for row_num, val in enumerate(col_values):
                if val == best_val:
                    worksheet.write(starting_row + row_num + 1, starting_col + col_num, val, good_format)
                elif val == second_best_val:
                    worksheet.write(starting_row + row_num + 1, starting_col + col_num, val, neutral_format)
                else:
                    worksheet.write(starting_row + row_num + 1, starting_col + col_num, val, base_format)
    worksheet.autofit()
