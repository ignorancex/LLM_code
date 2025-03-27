import xlrd
import os
import statistics
from xlwt import Workbook

path = "/home/mc1927/clara/batch_tests/algo2/1560B/"

wb = Workbook()
sheet1 = wb.add_sheet('test 1')
sheet1.write(0, 0, 'File')
sheet1.write(0, 1, 'Repairs')
sheet1.write(0, 2, 'Correct Repairs')
sheet1.write(0, 3, 'Partial Repairs')
sheet1.write(0, 4, 'Structure Mismatchs')
sheet1.write(0, 5, 'Max Locs Removed')
sheet1.write(0, 6, 'Min Locs Removed')
sheet1.write(0, 7, 'Max Locs Added')
sheet1.write(0, 8, 'Min Locs Added')
sheet1.write(0, 9, 'Parse Errors')
sheet1.write(0, 10, 'Max Percentage')
sheet1.write(0, 11, 'Min Percentage')
sheet1.write(0, 12, 'Average Percentage')
sheet1.write(0, 13, 'SD percentage')
sheet1.write(0, 14, 'Timeouts')
sheet1.write(0, 15, 'Total runs')
sheet1.write(0, 16, 'wrong test case')
sheet1.write(0, 17, 'avg matching score')
sheet1.write(0, 18, 'Repairs not needed')
i = 1

finalMatching = []
for probname in os.listdir(path):
    if '_3' not in probname:
        continue
    name = ''
    reps = 0
    correct_reps = 0
    mismatch = 0
    parse_err = 0
    partial_reps = 0
    runs = 0
    timeouts = 0
    percentage = []
    min_loc_added = 99999999
    max_loc_added = 0
    min_loc_rem = 99999999
    max_loc_rem = 0
    testcase = False
    matching_score = []
    tot = 0
    not_needed = 0

    prob_wb = xlrd.open_workbook(path+probname)
    prob_sheet = prob_wb.sheet_by_index(0)

    for j in range(1, prob_sheet.nrows):
        name = prob_sheet.cell_value(j, 1)
        name_c = prob_sheet.cell_value(j, 0)

        runs += 1
        if prob_sheet.cell_value(j, 7) != 'Yes':
            testcase = True
            continue
        if prob_sheet.cell_value(j, 3) == 'True':
            mismatch += 1
        else:
            if (prob_sheet.cell_value(j, 4) == 'Yes'):
                correct_reps += 1
                percentage += [float(prob_sheet.cell_value(j, 14))] if len(prob_sheet.cell_value(j, 14)) else []
                matching_score += [float(prob_sheet.cell_value(j, 13))] if len(prob_sheet.cell_value(j, 13)) else []
            elif (prob_sheet.cell_value(j, 4) == 'Not Needed'):
                not_needed += 1

        if (prob_sheet.cell_value(j, 11) != 0):
            if prob_sheet.cell_value(j, 9) == 'Add':
                min_loc_added = min(min_loc_added, int(prob_sheet.cell_value(j, 10)))
                max_loc_added = max(max_loc_added, int(prob_sheet.cell_value(j, 10)))
            elif prob_sheet.cell_value(j, 9) == 'Del':
                min_loc_rem = min(min_loc_rem, int(prob_sheet.cell_value(j, 10)))
                max_loc_rem = max(max_loc_rem, int(prob_sheet.cell_value(j, 10)))
        if (prob_sheet.cell_value(j, 8) == 'Yes'):
            timeouts += 1
        if (prob_sheet.cell_value(j, 2) == 'Yes'):
            reps += 1
        if (prob_sheet.cell_value(j, 4) == 'Partial'):
            partial_reps += 1
        if (prob_sheet.cell_value(j, 6) == 'Yes'):
            parse_err += 1
    sheet1.write(i, 0, name)
    sheet1.write(i, 1, reps)
    sheet1.write(i, 2, correct_reps)
    sheet1.write(i, 3, partial_reps)
    sheet1.write(i, 4, mismatch)
    sheet1.write(i, 5, max_loc_rem)
    sheet1.write(i, 6, min_loc_rem)
    sheet1.write(i, 7, max_loc_added)
    sheet1.write(i, 8, min_loc_added)
    sheet1.write(i, 9, parse_err)
    if len(percentage) > 0:
        sheet1.write(i, 10, max(percentage))
        sheet1.write(i, 11, min(percentage))
        sheet1.write(i, 12, statistics.mean(percentage))
        sheet1.write(i, 13, statistics.stdev(percentage)
                     if len(percentage) > 1 else 0)
    sheet1.write(i, 14, timeouts)
    sheet1.write(i, 15, runs)
    sheet1.write(i, 16, testcase)
    finalMatching += percentage
    score = statistics.mean(map(float, matching_score)) if len(matching_score) > 0 else 0
    sheet1.write(i, 17, float(score))
    sheet1.write(i, 18, not_needed)

    i += 1
wb.save('1560B_Algo2_Summary_3.xls')
