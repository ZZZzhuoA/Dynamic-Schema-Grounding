## Example 1

**DB:** `california_schools`

**Question:** What is the highest eligible free rate for K-12 students in the schools in Alameda County?

**Evidence:** Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `satscores` (table), score=3.5734
- `schools` (table), score=3.4070
- `frpm` (table), score=2.3784
- `schools.CDSCode` (column), score=1.7748
- `frpm.Low Grade` (column), score=1.5350
- `satscores.cds` (column), score=1.4820
- `frpm.School Code` (column), score=1.4698
- `frpm.Charter School Number` (column), score=1.2904
- `frpm.FRPM Count (K-12)` (column), score=1.1943
- `frpm.CDSCode` (column), score=1.1300
- `frpm.District Name` (column), score=1.0932
- `frpm.School Name` (column), score=1.0658
- `schools.District` (column), score=1.0627
- `schools.School` (column), score=1.0359
- `schools.AdmEmail2` (column), score=1.0206

## Example 2

**DB:** `california_schools`

**Question:** Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.

**Evidence:** Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `schools` (table), score=4.4629
- `satscores` (table), score=4.0844
- `frpm.Low Grade` (column), score=3.1807
- `frpm` (table), score=2.7442
- `satscores.cds` (column), score=2.5380
- `schools.District` (column), score=2.3156
- `schools.AdmEmail2` (column), score=2.1810
- `frpm.School Code` (column), score=2.1529
- `satscores.NumGE1500` (column), score=1.8852
- `frpm.2013-14 CALPADS Fall 1 Certification Status` (column), score=1.8383
- `schools.CDSCode` (column), score=1.7865
- `schools.SOC` (column), score=1.7514
- `satscores.AvgScrWrite` (column), score=1.6664
- `frpm.FRPM Count (K-12)` (column), score=1.6337
- `schools.Virtual` (column), score=1.6300

## Example 3

**DB:** `california_schools`

**Question:** Please list the zip code of all the charter schools in Fresno County Office of Education.

**Evidence:** Charter schools refers to `Charter School (Y/N)` = 1 in the table fprm

**Gold labels:**

- `frpm`
- `schools`
- `frpm.CDSCode`
- `schools.CDSCode`
- `schools.District`
- `schools.School`
- `schools.Zip`
- `schools.Charter`

**Top predictions:**

- `satscores` (table), score=3.2751
- `schools` (table), score=3.0127
- `frpm` (table), score=2.1378
- `schools.CDSCode` (column), score=1.5855
- `frpm.School Code` (column), score=1.1394
- `satscores.cds` (column), score=1.1157
- `frpm.Low Grade` (column), score=1.1064
- `frpm.CDSCode` (column), score=1.1049
- `frpm.Charter School Number` (column), score=1.0701
- `frpm.FRPM Count (K-12)` (column), score=1.0202
- `frpm.School Name` (column), score=0.9727
- `schools.School` (column), score=0.9379
- `frpm.County Name` (column), score=0.9271
- `frpm.District Name` (column), score=0.9165
- `frpm.School Type` (column), score=0.9086

## Example 4

**DB:** `california_schools`

**Question:** What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?

**Gold labels:**

- `frpm`
- `schools`
- `frpm.CDSCode`
- `schools.CDSCode`
- `schools.MailStreet`

**Top predictions:**

- `satscores` (table), score=1.8502
- `schools.SOCType` (column), score=1.7327
- `frpm.County Name` (column), score=1.0760
- `frpm.CDSCode` (column), score=0.9799
- `frpm` (table), score=0.7944
- `frpm.Enrollment (K-12)` (column), score=0.7941
- `schools` (table), score=0.7850
- `frpm.Academic Year` (column), score=0.6567
- `schools.CDSCode` (column), score=0.5003
- `frpm.School Name` (column), score=0.4989
- `schools.DOCType` (column), score=0.4914
- `frpm.Charter School (Y/N)` (column), score=0.3282
- `frpm.School Type` (column), score=0.1850
- `frpm.District Name` (column), score=0.0634
- `frpm.Charter School Number` (column), score=0.0053

## Example 5

**DB:** `california_schools`

**Question:** Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.

**Evidence:** Charter schools refers to `Charter School (Y/N)` = 1 in the frpm

**Gold labels:**

- `frpm`
- `schools`
- `frpm.CDSCode`
- `schools.CDSCode`
- `schools.School`
- `schools.Phone`
- `schools.OpenDate`
- `schools.Charter`

**Top predictions:**

- `satscores` (table), score=3.9757
- `schools` (table), score=3.6867
- `frpm` (table), score=3.0532
- `schools.CDSCode` (column), score=2.4197
- `frpm.CDSCode` (column), score=2.0465
- `frpm.County Name` (column), score=2.0090
- `frpm.Charter School Number` (column), score=1.9189
- `frpm.School Name` (column), score=1.9063
- `frpm.School Code` (column), score=1.8523
- `frpm.District Name` (column), score=1.8174
- `frpm.Charter School (Y/N)` (column), score=1.6855
- `satscores.cds` (column), score=1.6532
- `frpm.Enrollment (K-12)` (column), score=1.6326
- `frpm.School Type` (column), score=1.5885
- `frpm.Academic Year` (column), score=1.5814

## Example 6

**DB:** `california_schools`

**Question:** How many schools with an average score in Math greater than 400 in the SAT test are exclusively virtual?

**Evidence:** Exclusively virtual refers to Virtual = 'F'

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.AvgScrMath`
- `schools.CDSCode`
- `schools.School`
- `schools.Virtual`

**Top predictions:**

- `satscores` (table), score=4.4945
- `schools` (table), score=4.1439
- `frpm` (table), score=3.5585
- `schools.CDSCode` (column), score=2.8792
- `frpm.CDSCode` (column), score=2.5344
- `frpm.County Name` (column), score=2.5109
- `frpm.Charter School Number` (column), score=2.3816
- `frpm.School Name` (column), score=2.3362
- `frpm.District Name` (column), score=2.3311
- `frpm.Enrollment (K-12)` (column), score=2.2810
- `frpm.School Code` (column), score=2.2747
- `frpm.Academic Year` (column), score=2.1304
- `frpm.Charter School (Y/N)` (column), score=2.0974
- `frpm.FRPM Count (K-12)` (column), score=2.0276
- `satscores.cds` (column), score=1.9626

## Example 7

**DB:** `california_schools`

**Question:** Among the schools with the SAT test takers of over 500, please list the schools that are magnet schools or offer a magnet program.

**Evidence:** Magnet schools or offer a magnet program means that Magnet = 1

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.NumTstTakr`
- `schools.CDSCode`
- `schools.School`
- `schools.Magnet`

**Top predictions:**

- `satscores` (table), score=3.2705
- `schools` (table), score=2.5907
- `frpm` (table), score=2.2215
- `schools.CDSCode` (column), score=1.6554
- `frpm.CDSCode` (column), score=1.5115
- `frpm.County Name` (column), score=1.3571
- `frpm.School Name` (column), score=1.3252
- `frpm.Charter School Number` (column), score=1.1758
- `frpm.Charter School (Y/N)` (column), score=1.0496
- `frpm.District Name` (column), score=0.9755
- `frpm.School Type` (column), score=0.9371
- `frpm.Academic Year` (column), score=0.8998
- `frpm.School Code` (column), score=0.8764
- `frpm.Enrollment (K-12)` (column), score=0.8379
- `schools.SOCType` (column), score=0.8163

## Example 8

**DB:** `california_schools`

**Question:** What is the phone number of the school that has the highest number of test takers with an SAT score of over 1500?

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.NumGE1500`
- `schools.CDSCode`
- `schools.Phone`

**Top predictions:**

- `schools` (table), score=4.2927
- `satscores` (table), score=4.2860
- `frpm` (table), score=3.1405
- `schools.CDSCode` (column), score=2.4211
- `frpm.Low Grade` (column), score=2.3560
- `satscores.cds` (column), score=2.1851
- `frpm.School Code` (column), score=2.1494
- `frpm.FRPM Count (K-12)` (column), score=1.8475
- `frpm.Charter School Number` (column), score=1.8102
- `frpm.District Code` (column), score=1.6361
- `frpm.District Name` (column), score=1.5935
- `frpm.District Type` (column), score=1.5568
- `schools.District` (column), score=1.4701
- `frpm.FRPM Count (Ages 5-17)` (column), score=1.4579
- `frpm.CDSCode` (column), score=1.4455

## Example 9

**DB:** `california_schools`

**Question:** What is the number of SAT test takers of the schools with the highest FRPM count for K-12 students?

**Gold labels:**

- `frpm`
- `satscores`
- `frpm.CDSCode`
- `satscores.cds`
- `satscores.NumTstTakr`

**Top predictions:**

- `satscores` (table), score=2.9968
- `schools` (table), score=2.2484
- `frpm` (table), score=1.9526
- `frpm.CDSCode` (column), score=1.4582
- `schools.CDSCode` (column), score=1.4408
- `frpm.County Name` (column), score=1.4062
- `frpm.School Name` (column), score=1.2138
- `schools.SOCType` (column), score=1.0415
- `frpm.Enrollment (K-12)` (column), score=1.0211
- `frpm.Charter School (Y/N)` (column), score=0.9644
- `frpm.Academic Year` (column), score=0.9617
- `frpm.Charter School Number` (column), score=0.9561
- `frpm.District Name` (column), score=0.8570
- `frpm.School Type` (column), score=0.8411
- `frpm.FRPM Count (K-12)` (column), score=0.5831

## Example 10

**DB:** `california_schools`

**Question:** Among the schools with the average score in Math over 560 in the SAT test, how many schools are directly charter-funded?

**Gold labels:**

- `frpm`
- `satscores`
- `frpm.CDSCode`
- `satscores.cds`
- `satscores.AvgScrMath`

**Top predictions:**

- `satscores` (table), score=4.1481
- `schools` (table), score=4.1313
- `frpm` (table), score=3.1617
- `schools.CDSCode` (column), score=2.5542
- `frpm.Low Grade` (column), score=2.3318
- `frpm.School Code` (column), score=2.2622
- `satscores.cds` (column), score=2.2212
- `frpm.Charter School Number` (column), score=2.0927
- `frpm.FRPM Count (K-12)` (column), score=1.9060
- `frpm.District Name` (column), score=1.8766
- `frpm.District Code` (column), score=1.8373
- `frpm.District Type` (column), score=1.7719
- `frpm.School Name` (column), score=1.7555
- `frpm.CDSCode` (column), score=1.7072
- `frpm.School Type` (column), score=1.6258

## Example 11

**DB:** `california_schools`

**Question:** For the school with the highest average score in Reading in the SAT test, what is its FRPM count for students aged 5-17?

**Gold labels:**

- `frpm`
- `satscores`
- `frpm.CDSCode`
- `satscores.cds`
- `satscores.AvgScrRead`

**Top predictions:**

- `schools` (table), score=4.7227
- `satscores` (table), score=4.6339
- `frpm` (table), score=3.4801
- `frpm.Low Grade` (column), score=2.9989
- `satscores.cds` (column), score=2.7464
- `schools.CDSCode` (column), score=2.6780
- `frpm.School Code` (column), score=2.6013
- `frpm.FRPM Count (K-12)` (column), score=2.1144
- `frpm.Charter School Number` (column), score=2.0604
- `schools.District` (column), score=2.0220
- `frpm.District Code` (column), score=2.0216
- `schools.AdmEmail2` (column), score=1.9711
- `frpm.District Type` (column), score=1.8836
- `frpm.District Name` (column), score=1.8502
- `frpm.FRPM Count (Ages 5-17)` (column), score=1.7119

## Example 12

**DB:** `california_schools`

**Question:** Please list the codes of the schools with a total enrollment of over 500.

**Evidence:** Total enrollment can be represented by `Enrollment (K-12)` + `Enrollment (Ages 5-17)`

**Gold labels:**

- `frpm`
- `schools`
- `frpm.CDSCode`
- `schools.CDSCode`

**Top predictions:**

- `satscores` (table), score=3.9655
- `schools` (table), score=3.3112
- `frpm` (table), score=3.1917
- `frpm.County Name` (column), score=2.7339
- `schools.CDSCode` (column), score=2.5625
- `frpm.CDSCode` (column), score=2.5527
- `frpm.Enrollment (K-12)` (column), score=2.5218
- `frpm.School Name` (column), score=2.2721
- `frpm.Academic Year` (column), score=2.2605
- `frpm.District Name` (column), score=2.0269
- `frpm.Charter School Number` (column), score=1.9777
- `frpm.Charter School (Y/N)` (column), score=1.9138
- `frpm.School Type` (column), score=1.6433
- `frpm.County Code` (column), score=1.5631
- `frpm.FRPM Count (K-12)` (column), score=1.5324

## Example 13

**DB:** `california_schools`

**Question:** Among the schools with an SAT excellence rate of over 0.3, what is the highest eligible free rate for students aged 5-17?

**Evidence:** Excellence rate = NumGE1500 / NumTstTakr; Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`

**Gold labels:**

- `frpm`
- `satscores`
- `frpm.CDSCode`
- `satscores.cds`
- `satscores.NumTstTakr`
- `satscores.NumGE1500`

**Top predictions:**

- `schools` (table), score=3.9140
- `satscores` (table), score=3.4435
- `frpm.Low Grade` (column), score=2.9740
- `satscores.AvgScrWrite` (column), score=2.1824
- `schools.District` (column), score=2.1283
- `satscores.cds` (column), score=1.9610
- `satscores.NumGE1500` (column), score=1.9430
- `schools.AdmEmail2` (column), score=1.8875
- `schools.DOC` (column), score=1.8874
- `schools.SOC` (column), score=1.8493
- `frpm.2013-14 CALPADS Fall 1 Certification Status` (column), score=1.8316
- `frpm` (table), score=1.8200
- `schools.Virtual` (column), score=1.4072
- `schools.State` (column), score=1.3094
- `frpm.School Code` (column), score=1.2935

## Example 14

**DB:** `california_schools`

**Question:** Please list the phone numbers of the schools with the top 3 SAT excellence rate.

**Evidence:** Excellence rate = NumGE1500 / NumTstTakr

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.NumTstTakr`
- `satscores.NumGE1500`
- `schools.CDSCode`
- `schools.Phone`

**Top predictions:**

- `satscores` (table), score=3.4014
- `schools` (table), score=3.2975
- `frpm` (table), score=2.0645
- `frpm.Low Grade` (column), score=1.6842
- `satscores.cds` (column), score=1.4475
- `schools.CDSCode` (column), score=1.2803
- `frpm.School Code` (column), score=1.2014
- `schools.District` (column), score=1.1450
- `schools.AdmEmail2` (column), score=1.0534
- `satscores.NumGE1500` (column), score=1.0062
- `schools.School` (column), score=0.9170
- `frpm.2013-14 CALPADS Fall 1 Certification Status` (column), score=0.9078
- `schools.Virtual` (column), score=0.8594
- `frpm.FRPM Count (K-12)` (column), score=0.8540
- `schools.DOC` (column), score=0.8024

## Example 15

**DB:** `california_schools`

**Question:** List the top five schools, by descending order, from the highest to the lowest, the most number of Enrollment (Ages 5-17). Please give their NCES school identification number.

**Gold labels:**

- `frpm`
- `schools`
- `frpm.CDSCode`
- `schools.CDSCode`
- `schools.NCESSchool`

**Top predictions:**

- `satscores` (table), score=2.8232
- `schools` (table), score=2.2221
- `frpm` (table), score=1.6410
- `schools.CDSCode` (column), score=1.1882
- `frpm.CDSCode` (column), score=0.9395
- `frpm.School Name` (column), score=0.7817
- `frpm.Charter School Number` (column), score=0.7464
- `frpm.County Name` (column), score=0.6644
- `schools.SOCType` (column), score=0.6637
- `frpm.School Code` (column), score=0.5800
- `frpm.Charter School (Y/N)` (column), score=0.5502
- `frpm.School Type` (column), score=0.5283
- `frpm.District Name` (column), score=0.4879
- `schools.OpenDate` (column), score=0.4375
- `satscores.cds` (column), score=0.4064

## Example 16

**DB:** `california_schools`

**Question:** Which active district has the highest average score in Reading?

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.AvgScrRead`
- `schools.CDSCode`
- `schools.StatusType`
- `schools.District`

**Top predictions:**

- `schools` (table), score=4.8388
- `satscores` (table), score=4.7750
- `frpm` (table), score=3.6067
- `frpm.Low Grade` (column), score=3.0271
- `schools.CDSCode` (column), score=2.8182
- `satscores.cds` (column), score=2.7995
- `frpm.School Code` (column), score=2.6867
- `frpm.Charter School Number` (column), score=2.2129
- `frpm.FRPM Count (K-12)` (column), score=2.1423
- `frpm.District Code` (column), score=2.0702
- `frpm.District Name` (column), score=2.0166
- `schools.District` (column), score=1.9607
- `schools.AdmEmail2` (column), score=1.9201
- `frpm.District Type` (column), score=1.9085
- `frpm.FRPM Count (Ages 5-17)` (column), score=1.7134

## Example 17

**DB:** `california_schools`

**Question:** How many schools in merged Alameda have number of test takers less than 100?

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.NumTstTakr`
- `schools.CDSCode`
- `schools.StatusType`
- `schools.County`

**Top predictions:**

- `schools.SOCType` (column), score=1.7458
- `satscores` (table), score=1.2844
- `frpm.CDSCode` (column), score=0.6199
- `frpm.County Name` (column), score=0.5971
- `schools.DOCType` (column), score=0.2890
- `frpm.Enrollment (K-12)` (column), score=0.2277
- `frpm` (table), score=0.2000
- `frpm.Academic Year` (column), score=0.1401
- `frpm.School Name` (column), score=0.0933
- `schools` (table), score=0.0283
- `schools.CDSCode` (column), score=-0.1185
- `frpm.Charter School (Y/N)` (column), score=-0.1349
- `frpm.School Type` (column), score=-0.3073
- `schools.EdOpsName` (column), score=-0.3673
- `frpm.District Name` (column), score=-0.3940

## Example 18

**DB:** `california_schools`

**Question:** Rank schools by their average score in Writing where the score is greater than 499, showing their charter numbers.

**Evidence:** Valid charter number means the number is not null

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.AvgScrWrite`
- `schools.CDSCode`
- `schools.CharterNum`

**Top predictions:**

- `satscores` (table), score=3.6580
- `schools` (table), score=2.9174
- `frpm` (table), score=2.7846
- `frpm.County Name` (column), score=2.5846
- `frpm.Enrollment (K-12)` (column), score=2.4183
- `frpm.CDSCode` (column), score=2.3429
- `schools.CDSCode` (column), score=2.1392
- `frpm.Academic Year` (column), score=2.1339
- `frpm.School Name` (column), score=1.9559
- `schools.SOCType` (column), score=1.7384
- `schools.AdmEmail1` (column), score=1.7069
- `frpm.District Name` (column), score=1.6751
- `frpm.Charter School (Y/N)` (column), score=1.6371
- `frpm.Charter School Number` (column), score=1.5417
- `frpm.County Code` (column), score=1.3728

## Example 19

**DB:** `california_schools`

**Question:** How many schools in Fresno (directly funded) have number of test takers not more than 250?

**Gold labels:**

- `frpm`
- `satscores`
- `frpm.CDSCode`
- `satscores.cds`
- `satscores.NumTstTakr`

**Top predictions:**

- `schools.SOCType` (column), score=1.7410
- `satscores` (table), score=1.4014
- `frpm.County Name` (column), score=0.7355
- `frpm.CDSCode` (column), score=0.6975
- `schools.DOCType` (column), score=0.4748
- `frpm.Enrollment (K-12)` (column), score=0.4401
- `frpm.Academic Year` (column), score=0.3561
- `frpm` (table), score=0.3495
- `schools` (table), score=0.2768
- `frpm.School Name` (column), score=0.1489
- `schools.CDSCode` (column), score=0.0495
- `frpm.Charter School (Y/N)` (column), score=0.0210
- `frpm.School Type` (column), score=-0.1257
- `schools.EdOpsName` (column), score=-0.1316
- `schools.AdmLName2` (column), score=-0.1432

## Example 20

**DB:** `california_schools`

**Question:** What is the phone number of the school that has the highest average score in Math?

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.AvgScrMath`
- `schools.CDSCode`
- `schools.Phone`

**Top predictions:**

- `schools` (table), score=5.1536
- `satscores` (table), score=5.0944
- `frpm` (table), score=3.9911
- `schools.CDSCode` (column), score=3.2358
- `frpm.Low Grade` (column), score=3.1118
- `frpm.School Code` (column), score=2.9544
- `satscores.cds` (column), score=2.9362
- `frpm.Charter School Number` (column), score=2.7080
- `frpm.District Name` (column), score=2.6156
- `frpm.District Code` (column), score=2.5738
- `frpm.District Type` (column), score=2.4746
- `frpm.FRPM Count (K-12)` (column), score=2.4727
- `frpm.School Name` (column), score=2.3089
- `frpm.CDSCode` (column), score=2.1689
- `frpm.School Type` (column), score=2.0732

