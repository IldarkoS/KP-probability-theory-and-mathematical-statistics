import xlwt
import numpy as np


n = 40
book = xlwt.Workbook(encoding="utf-8")

sheet1 = book.add_sheet("Python Sheet 1")

cols = ["A"]
txt = np.random.normal(0, np.sqrt(1.6), n)

for i in range(n):
      sheet1.write(i, 0, txt[i])

book.save("spreadsheet.xls")


# book = xlwt.Workbook(encoding="utf-8")
#
# sheet1 = book.add_sheet("Python Sheet 1")
#
# cols = ["A"]
# txt = np.random.uniform(-3*np.sqrt(1.6), 3*np.sqrt(1.6), n)
#
# for i in range(n):
#       sheet1.write(i, 0, txt[i])
#
# book.save("spreadsheet.xls")