## Example 1

**DB:** `california_schools`

**Question:** What is the highest eligible free rate for K-12 students in the schools in Alameda County?

**Evidence:** Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `frpm.Free Meal Count (K-12)` (column), score=3.8751
- `frpm.Enrollment (K-12)` (column), score=3.4769
- `schools.County` (column), score=3.3114
- `schools` (table), score=2.0541
- `schools.CDSCode` (column), score=0.8039
- `frpm` (table), score=-0.2198
- `satscores` (table), score=-0.3208
- `schools.AdmEmail1` (column), score=-0.4729
- `schools.District` (column), score=-0.5331
- `schools.AdmEmail2` (column), score=-0.7514
- `schools.School` (column), score=-0.8050
- `frpm.Percent (%) Eligible Free (K-12)` (column), score=-0.8126
- `frpm.FRPM Count (K-12)` (column), score=-0.8770
- `schools.EILCode` (column), score=-1.1899
- `schools.Ext` (column), score=-1.1899

## Example 2

**DB:** `california_schools`

**Question:** Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.

**Evidence:** Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `frpm.Free Meal Count (Ages 5-17)` (column), score=3.8174
- `frpm.Enrollment (Ages 5-17)` (column), score=3.3979
- `schools` (table), score=2.0219
- `schools.CDSCode` (column), score=0.7083
- `frpm` (table), score=-0.2855
- `satscores` (table), score=-0.3193
- `schools.AdmEmail1` (column), score=-0.5688
- `schools.District` (column), score=-0.6645
- `frpm.FRPM Count (Ages 5-17)` (column), score=-0.7006
- `frpm.Percent (%) Eligible Free (Ages 5-17)` (column), score=-0.8000
- `schools.AdmEmail2` (column), score=-0.8878
- `schools.School` (column), score=-0.9259
- `schools.Ext` (column), score=-1.2983
- `schools.EILCode` (column), score=-1.2983
- `frpm.Percent (%) Eligible FRPM (Ages 5-17)` (column), score=-1.3216

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

- `schools.Charter` (column), score=4.0640
- `schools` (table), score=4.0035
- `schools.School` (column), score=3.8657
- `schools.County` (column), score=3.7662
- `frpm.Charter School (Y/N)` (column), score=3.6354
- `schools.Zip` (column), score=2.7428
- `frpm` (table), score=1.6243
- `satscores` (table), score=1.3490
- `schools.CDSCode` (column), score=1.3487
- `frpm.School Code` (column), score=0.7327
- `schools.District` (column), score=0.2028
- `schools.AdmEmail1` (column), score=0.0855
- `schools.CharterNum` (column), score=0.0130
- `schools.AdmEmail2` (column), score=-0.1021
- `schools.Ext` (column), score=-0.4874

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

- `schools.School` (column), score=2.1143
- `frpm` (table), score=2.1134
- `frpm.FRPM Count (K-12)` (column), score=1.1168
- `schools.Street` (column), score=1.0950
- `frpm.School Name` (column), score=0.2891
- `frpm.School Code` (column), score=0.2202
- `frpm.School Type` (column), score=0.0514
- `schools` (table), score=-0.2450
- `frpm.CDSCode` (column), score=-0.2646
- `satscores` (table), score=-0.2953
- `frpm.Enrollment (K-12)` (column), score=-0.3070
- `frpm.Charter School Number` (column), score=-0.3101
- `frpm.District Name` (column), score=-0.3494
- `frpm.Free Meal Count (K-12)` (column), score=-0.8348
- `frpm.District Type` (column), score=-0.8514

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

- `frpm.Charter School (Y/N)` (column), score=4.6060
- `schools.Charter` (column), score=4.2066
- `schools.School` (column), score=3.9570
- `schools.Phone` (column), score=2.9820
- `schools` (table), score=2.7447
- `frpm` (table), score=2.3313
- `frpm.Charter School Number` (column), score=1.4955
- `schools.CDSCode` (column), score=1.4620
- `frpm.School Code` (column), score=0.8971
- `frpm.School Name` (column), score=0.8294
- `frpm.School Type` (column), score=0.7736
- `schools.District` (column), score=0.4129
- `schools.AdmEmail1` (column), score=0.2811
- `schools.CharterNum` (column), score=0.2583
- `frpm.CDSCode` (column), score=0.2575

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

- `schools.Virtual` (column), score=3.3928
- `schools` (table), score=2.0568
- `schools.CDSCode` (column), score=0.3485
- `satscores` (table), score=-0.2137
- `frpm` (table), score=-0.3710
- `schools.AdmEmail1` (column), score=-0.9705
- `schools.District` (column), score=-1.1241
- `schools.School` (column), score=-1.3493
- `schools.AdmEmail2` (column), score=-1.3652
- `schools.EILCode` (column), score=-1.6881
- `schools.Ext` (column), score=-1.6881
- `schools.EILName` (column), score=-1.7086
- `schools.Latitude` (column), score=-1.8890
- `schools.AdmLName2` (column), score=-1.9329
- `schools.GSserved` (column), score=-1.9746

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

- `schools.Magnet` (column), score=4.0059
- `schools` (table), score=2.7514
- `schools.CDSCode` (column), score=1.3439
- `schools.District` (column), score=0.1977
- `schools.AdmEmail1` (column), score=0.1295
- `schools.School` (column), score=-0.0755
- `schools.AdmEmail2` (column), score=-0.1234
- `frpm` (table), score=-0.2537
- `satscores` (table), score=-0.3486
- `schools.EILCode` (column), score=-0.4644
- `schools.Ext` (column), score=-0.4644
- `schools.EILName` (column), score=-0.5388
- `schools.Latitude` (column), score=-0.6442
- `schools.GSserved` (column), score=-0.7506
- `schools.Virtual` (column), score=-0.8883

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

- `schools.School` (column), score=2.0116
- `schools.Phone` (column), score=-0.0802
- `satscores` (table), score=-0.1930
- `frpm` (table), score=-0.1988
- `schools` (table), score=-0.2039
- `schools.CDSCode` (column), score=-1.3602
- `frpm.Charter School Number` (column), score=-1.5533
- `satscores.cds` (column), score=-1.6087
- `frpm.School Name` (column), score=-1.7803
- `frpm.School Code` (column), score=-1.9593
- `frpm.School Type` (column), score=-2.2560
- `frpm.District Name` (column), score=-2.3916
- `schools.AdmEmail1` (column), score=-2.4992
- `frpm.CDSCode` (column), score=-2.5224
- `schools.District` (column), score=-2.7782

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

- `frpm` (table), score=2.1086
- `schools` (table), score=2.0734
- `frpm.FRPM Count (K-12)` (column), score=0.9801
- `schools.CDSCode` (column), score=0.7081
- `satscores` (table), score=-0.2775
- `frpm.CDSCode` (column), score=-0.3925
- `frpm.Enrollment (K-12)` (column), score=-0.4343
- `frpm.Charter School Number` (column), score=-0.4482
- `frpm.District Name` (column), score=-0.4555
- `schools.AdmEmail1` (column), score=-0.5617
- `schools.District` (column), score=-0.6454
- `frpm.School Name` (column), score=-0.7124
- `frpm.School Code` (column), score=-0.8266
- `schools.AdmEmail2` (column), score=-0.8682
- `schools.School` (column), score=-0.8989

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

- `schools.Charter` (column), score=2.9667
- `schools` (table), score=2.0629
- `schools.CDSCode` (column), score=0.5877
- `satscores` (table), score=-0.2403
- `frpm` (table), score=-0.2894
- `schools.AdmEmail1` (column), score=-0.6734
- `schools.District` (column), score=-0.8115
- `schools.AdmEmail2` (column), score=-1.0430
- `schools.School` (column), score=-1.0456
- `schools.EILCode` (column), score=-1.4087
- `schools.Ext` (column), score=-1.4087
- `schools.EILName` (column), score=-1.4627
- `schools.CharterNum` (column), score=-1.5681
- `schools.Latitude` (column), score=-1.6091
- `schools.GSserved` (column), score=-1.6998

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

- `schools.School` (column), score=2.1668
- `frpm` (table), score=2.1455
- `frpm.School Name` (column), score=0.3395
- `frpm.School Code` (column), score=0.2307
- `frpm.School Type` (column), score=0.0995
- `schools` (table), score=-0.1788
- `satscores` (table), score=-0.2371
- `frpm.CDSCode` (column), score=-0.2546
- `frpm.Charter School Number` (column), score=-0.2674
- `frpm.District Name` (column), score=-0.3006
- `frpm.FRPM Count (Ages 5-17)` (column), score=-0.6133
- `frpm.District Type` (column), score=-0.8005
- `frpm.District Code` (column), score=-0.8847
- `frpm.Charter School (Y/N)` (column), score=-0.9906
- `schools.CDSCode` (column), score=-1.1661

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

- `frpm.Enrollment (K-12)` (column), score=2.7007
- `frpm.Enrollment (Ages 5-17)` (column), score=2.6196
- `schools` (table), score=2.1398
- `schools.CDSCode` (column), score=0.2415
- `satscores` (table), score=-0.0653
- `frpm` (table), score=-0.3688
- `schools.AdmEmail1` (column), score=-1.0104
- `schools.District` (column), score=-1.2272
- `schools.School` (column), score=-1.4467
- `schools.AdmEmail2` (column), score=-1.4806
- `schools.Ext` (column), score=-1.7138
- `schools.EILCode` (column), score=-1.7138
- `schools.EILName` (column), score=-1.7474
- `schools.AdmLName2` (column), score=-1.9021
- `schools.Latitude` (column), score=-1.9695

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

- `frpm.Free Meal Count (Ages 5-17)` (column), score=3.8703
- `frpm.Enrollment (Ages 5-17)` (column), score=3.4763
- `satscores.NumTstTakr` (column), score=3.0241
- `satscores.NumGE1500` (column), score=2.7703
- `schools` (table), score=2.0263
- `schools.CDSCode` (column), score=0.7781
- `frpm` (table), score=-0.2264
- `satscores` (table), score=-0.3480
- `schools.AdmEmail1` (column), score=-0.5002
- `schools.District` (column), score=-0.5441
- `frpm.FRPM Count (Ages 5-17)` (column), score=-0.5922
- `frpm.Percent (%) Eligible Free (Ages 5-17)` (column), score=-0.6763
- `schools.AdmEmail2` (column), score=-0.7580
- `schools.School` (column), score=-0.8310
- `schools.CharterNum` (column), score=-1.0018

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

- `satscores.NumTstTakr` (column), score=2.8867
- `satscores.NumGE1500` (column), score=2.6381
- `schools` (table), score=2.0747
- `schools.Phone` (column), score=2.0720
- `schools.CDSCode` (column), score=0.6923
- `frpm` (table), score=-0.2302
- `satscores` (table), score=-0.2405
- `schools.AdmEmail1` (column), score=-0.6098
- `schools.District` (column), score=-0.7041
- `schools.AdmEmail2` (column), score=-0.9332
- `schools.School` (column), score=-0.9602
- `schools.CharterNum` (column), score=-1.1975
- `schools.EILCode` (column), score=-1.3343
- `schools.Ext` (column), score=-1.3343
- `schools.EILName` (column), score=-1.3826

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

- `schools.School` (column), score=3.4214
- `frpm.Enrollment (Ages 5-17)` (column), score=2.5170
- `schools` (table), score=1.9783
- `schools.CDSCode` (column), score=0.7340
- `frpm` (table), score=-0.2640
- `satscores` (table), score=-0.4351
- `schools.AdmEmail1` (column), score=-0.5312
- `schools.District` (column), score=-0.5711
- `schools.AdmEmail2` (column), score=-0.7832
- `schools.EILCode` (column), score=-1.2708
- `schools.Ext` (column), score=-1.2708
- `schools.EILName` (column), score=-1.3695
- `satscores.cds` (column), score=-1.3784
- `frpm.Charter School Number` (column), score=-1.3979
- `schools.Latitude` (column), score=-1.4471

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

- `schools.District` (column), score=2.1955
- `frpm` (table), score=-0.1146
- `schools` (table), score=-0.1323
- `satscores` (table), score=-0.1416
- `schools.CDSCode` (column), score=-1.1651
- `frpm.District Name` (column), score=-1.3418
- `satscores.cds` (column), score=-1.4989
- `frpm.District Type` (column), score=-1.8420
- `frpm.District Code` (column), score=-1.9496
- `schools.AdmEmail1` (column), score=-2.3173
- `frpm.CDSCode` (column), score=-2.3316
- `frpm.School Name` (column), score=-2.4786
- `schools.AdmEmail2` (column), score=-2.6032
- `frpm.School Code` (column), score=-2.6366
- `frpm.Charter School Number` (column), score=-2.8406

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

- `schools` (table), score=2.0666
- `schools.CDSCode` (column), score=0.6060
- `frpm` (table), score=-0.2223
- `satscores` (table), score=-0.2665
- `schools.AdmEmail1` (column), score=-0.6857
- `schools.District` (column), score=-0.7728
- `schools.AdmEmail2` (column), score=-0.9973
- `schools.School` (column), score=-1.0196
- `schools.Ext` (column), score=-1.3989
- `schools.EILCode` (column), score=-1.3989
- `schools.EILName` (column), score=-1.4515
- `schools.Latitude` (column), score=-1.5816
- `satscores.cds` (column), score=-1.6661
- `schools.GSserved` (column), score=-1.6802
- `schools.AdmLName2` (column), score=-1.7463

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

- `schools.Charter` (column), score=3.3644
- `schools` (table), score=2.1868
- `schools.CDSCode` (column), score=0.4294
- `satscores` (table), score=-0.0068
- `frpm` (table), score=-0.2458
- `schools.AdmEmail1` (column), score=-0.8516
- `schools.District` (column), score=-1.0227
- `schools.School` (column), score=-1.2470
- `schools.AdmEmail2` (column), score=-1.2719
- `schools.CharterNum` (column), score=-1.2887
- `frpm.Charter School Number` (column), score=-1.3111
- `schools.EILCode` (column), score=-1.5357
- `schools.Ext` (column), score=-1.5357
- `schools.EILName` (column), score=-1.5683
- `schools.Latitude` (column), score=-1.7658

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

- `schools` (table), score=2.1081
- `schools.CDSCode` (column), score=0.9631
- `frpm` (table), score=-0.1185
- `satscores` (table), score=-0.2522
- `schools.District` (column), score=-0.2660
- `schools.AdmEmail1` (column), score=-0.2674
- `schools.AdmEmail2` (column), score=-0.4780
- `schools.School` (column), score=-0.5411
- `schools.Ext` (column), score=-0.9372
- `schools.EILCode` (column), score=-0.9372
- `schools.EILName` (column), score=-1.0258
- `schools.Latitude` (column), score=-1.1074
- `schools.GSserved` (column), score=-1.2195
- `satscores.cds` (column), score=-1.2431
- `schools.AdmLName2` (column), score=-1.3419

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

- `schools.School` (column), score=2.0269
- `schools.Phone` (column), score=-0.0880
- `satscores` (table), score=-0.1872
- `schools` (table), score=-0.1874
- `frpm` (table), score=-0.1901
- `schools.CDSCode` (column), score=-1.3379
- `frpm.Charter School Number` (column), score=-1.5459
- `satscores.cds` (column), score=-1.6109
- `frpm.School Name` (column), score=-1.7656
- `frpm.School Code` (column), score=-1.9518
- `frpm.School Type` (column), score=-2.2446
- `frpm.District Name` (column), score=-2.3822
- `schools.AdmEmail1` (column), score=-2.4883
- `frpm.CDSCode` (column), score=-2.5161
- `schools.District` (column), score=-2.7685

