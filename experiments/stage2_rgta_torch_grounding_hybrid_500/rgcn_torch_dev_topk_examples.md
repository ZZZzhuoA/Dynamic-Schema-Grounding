## Example 1

**DB:** `california_schools`

**Question:** What is the highest eligible free rate for K-12 students in the schools in Alameda County?

**Evidence:** Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `frpm.Free Meal Count (K-12)` (column), score=3.3953
- `frpm.Enrollment (K-12)` (column), score=3.1047
- `schools.County` (column), score=3.0652
- `schools` (table), score=1.0678
- `schools.CDSCode` (column), score=0.8410
- `schools.District` (column), score=-0.0429
- `frpm.FRPM Count (K-12)` (column), score=-0.1610
- `schools.School` (column), score=-0.1614
- `frpm` (table), score=-0.2605
- `schools.AdmEmail2` (column), score=-0.2928
- `frpm.Percent (%) Eligible Free (K-12)` (column), score=-0.3535
- `schools.AdmEmail1` (column), score=-0.4341
- `schools.EILName` (column), score=-0.6078
- `satscores` (table), score=-0.6291
- `schools.EILCode` (column), score=-0.6663

## Example 2

**DB:** `california_schools`

**Question:** Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.

**Evidence:** Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `frpm.Free Meal Count (Ages 5-17)` (column), score=3.2312
- `frpm.Enrollment (Ages 5-17)` (column), score=2.8907
- `schools` (table), score=1.0812
- `schools.CDSCode` (column), score=0.6199
- `frpm` (table), score=-0.2281
- `frpm.FRPM Count (Ages 5-17)` (column), score=-0.5784
- `satscores` (table), score=-0.6257
- `schools.District` (column), score=-0.8365
- `frpm.Percent (%) Eligible Free (Ages 5-17)` (column), score=-0.8709
- `schools.School` (column), score=-0.9264
- `schools.AdmEmail2` (column), score=-1.0603
- `schools.AdmEmail1` (column), score=-1.1554
- `schools.EILName` (column), score=-1.3194
- `schools.EILCode` (column), score=-1.3672
- `schools.Ext` (column), score=-1.3672

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

- `schools.Charter` (column), score=3.3573
- `schools.School` (column), score=3.2880
- `schools.County` (column), score=3.1578
- `frpm.Charter School (Y/N)` (column), score=3.0920
- `schools.Zip` (column), score=2.7517
- `schools` (table), score=2.3253
- `schools.CDSCode` (column), score=0.8013
- `frpm` (table), score=0.5673
- `frpm.School Code` (column), score=0.5438
- `satscores` (table), score=0.2910
- `schools.CharterNum` (column), score=0.1184
- `schools.District` (column), score=0.0661
- `schools.AdmEmail2` (column), score=-0.2906
- `schools.EILName` (column), score=-0.4374
- `schools.AdmEmail1` (column), score=-0.4403

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

- `schools.School` (column), score=2.4580
- `schools.Street` (column), score=1.8153
- `frpm.FRPM Count (K-12)` (column), score=1.6102
- `frpm` (table), score=1.1569
- `frpm.School Type` (column), score=0.7540
- `frpm.School Name` (column), score=0.6883
- `frpm.School Code` (column), score=0.5650
- `frpm.Enrollment (K-12)` (column), score=0.3777
- `frpm.Charter School Number` (column), score=0.3460
- `schools` (table), score=0.0048
- `schools.CDSCode` (column), score=0.0010
- `frpm.District Name` (column), score=-0.1342
- `frpm.District Type` (column), score=-0.2400
- `frpm.Free Meal Count (K-12)` (column), score=-0.3219
- `frpm.CDSCode` (column), score=-0.3696

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

- `frpm.Charter School (Y/N)` (column), score=3.7665
- `schools.Charter` (column), score=3.5054
- `schools.School` (column), score=3.4181
- `schools.Phone` (column), score=3.0441
- `frpm.Charter School Number` (column), score=1.6345
- `schools` (table), score=1.5766
- `frpm` (table), score=1.1363
- `frpm.School Type` (column), score=1.0530
- `schools.CDSCode` (column), score=0.9908
- `frpm.School Name` (column), score=0.8506
- `frpm.School Code` (column), score=0.8135
- `schools.District` (column), score=0.4512
- `schools.CharterNum` (column), score=0.3498
- `schools.AdmEmail2` (column), score=0.0902
- `schools.AdmEmail1` (column), score=-0.0375

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

- `schools.Virtual` (column), score=3.1877
- `schools` (table), score=1.1313
- `schools.CDSCode` (column), score=0.7070
- `frpm` (table), score=-0.1833
- `satscores` (table), score=-0.5745
- `schools.District` (column), score=-0.6838
- `schools.School` (column), score=-0.8035
- `schools.AdmEmail2` (column), score=-0.9039
- `schools.AdmEmail1` (column), score=-1.0096
- `schools.EILName` (column), score=-1.1572
- `schools.Ext` (column), score=-1.2162
- `schools.EILCode` (column), score=-1.2162
- `schools.OpenDate` (column), score=-1.4439
- `schools.SOC` (column), score=-1.4741
- `schools.Latitude` (column), score=-1.4793

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

- `schools.Magnet` (column), score=3.5270
- `schools` (table), score=1.4171
- `schools.CDSCode` (column), score=1.0008
- `schools.District` (column), score=0.7133
- `schools.School` (column), score=0.6127
- `schools.AdmEmail2` (column), score=0.3643
- `schools.AdmEmail1` (column), score=0.2157
- `schools.EILName` (column), score=0.1688
- `schools.Ext` (column), score=0.1121
- `schools.EILCode` (column), score=0.1121
- `schools.SOC` (column), score=-0.0621
- `schools.Latitude` (column), score=-0.0765
- `schools.OpenDate` (column), score=-0.1459
- `schools.State` (column), score=-0.2165
- `schools.Virtual` (column), score=-0.2558

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

- `schools.School` (column), score=2.4841
- `schools.Phone` (column), score=0.7995
- `schools.CDSCode` (column), score=0.1104
- `schools` (table), score=0.0373
- `frpm` (table), score=-0.0190
- `satscores` (table), score=-0.3698
- `frpm.Charter School Number` (column), score=-0.4891
- `satscores.cds` (column), score=-0.9517
- `frpm.School Name` (column), score=-1.0778
- `frpm.School Type` (column), score=-1.2148
- `frpm.School Code` (column), score=-1.2318
- `frpm.District Name` (column), score=-1.8629
- `schools.District` (column), score=-2.0553
- `frpm.Charter School (Y/N)` (column), score=-2.0824
- `schools.AdmEmail2` (column), score=-2.0915

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

- `frpm.FRPM Count (K-12)` (column), score=1.6069
- `schools` (table), score=1.2261
- `frpm` (table), score=1.1220
- `schools.CDSCode` (column), score=0.9112
- `frpm.Charter School Number` (column), score=0.3441
- `frpm.Enrollment (K-12)` (column), score=0.3440
- `schools.District` (column), score=-0.1082
- `frpm.District Name` (column), score=-0.1528
- `schools.School` (column), score=-0.2331
- `frpm.District Type` (column), score=-0.2336
- `frpm.School Name` (column), score=-0.2420
- `frpm.School Code` (column), score=-0.3345
- `schools.AdmEmail2` (column), score=-0.3362
- `frpm.Free Meal Count (K-12)` (column), score=-0.3429
- `frpm.School Type` (column), score=-0.3496

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

- `schools.Charter` (column), score=2.9132
- `schools` (table), score=1.3093
- `schools.CDSCode` (column), score=0.8727
- `frpm` (table), score=-0.0035
- `satscores` (table), score=-0.3925
- `schools.District` (column), score=-0.4129
- `schools.School` (column), score=-0.5583
- `schools.AdmEmail2` (column), score=-0.6453
- `schools.AdmEmail1` (column), score=-0.7395
- `schools.EILName` (column), score=-0.8847
- `schools.Ext` (column), score=-0.9623
- `schools.EILCode` (column), score=-0.9623
- `schools.CharterNum` (column), score=-1.1218
- `schools.OpenDate` (column), score=-1.2033
- `schools.SOC` (column), score=-1.2673

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

- `schools.School` (column), score=2.5066
- `frpm` (table), score=1.2852
- `frpm.School Type` (column), score=0.8032
- `frpm.School Name` (column), score=0.7900
- `frpm.School Code` (column), score=0.5690
- `frpm.Charter School Number` (column), score=0.4271
- `schools` (table), score=0.1227
- `schools.CDSCode` (column), score=0.1041
- `frpm.District Name` (column), score=0.0007
- `frpm.FRPM Count (Ages 5-17)` (column), score=-0.0910
- `frpm.District Type` (column), score=-0.1633
- `frpm.CDSCode` (column), score=-0.2584
- `satscores` (table), score=-0.3087
- `frpm.District Code` (column), score=-0.4072
- `frpm.Charter School (Y/N)` (column), score=-0.4844

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

- `frpm.Enrollment (K-12)` (column), score=2.8398
- `frpm.Enrollment (Ages 5-17)` (column), score=2.4706
- `schools` (table), score=1.5428
- `schools.CDSCode` (column), score=0.6639
- `frpm` (table), score=0.2113
- `satscores` (table), score=-0.1811
- `schools.District` (column), score=-1.2907
- `schools.School` (column), score=-1.4590
- `schools.AdmEmail2` (column), score=-1.4930
- `schools.AdmEmail1` (column), score=-1.5030
- `schools.EILName` (column), score=-1.5888
- `schools.Ext` (column), score=-1.6532
- `schools.EILCode` (column), score=-1.6532
- `schools.AdmLName2` (column), score=-1.9744
- `schools.OpenDate` (column), score=-2.0651

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

- `frpm.Free Meal Count (Ages 5-17)` (column), score=3.1108
- `frpm.Enrollment (Ages 5-17)` (column), score=2.8668
- `satscores.NumTstTakr` (column), score=2.5644
- `satscores.NumGE1500` (column), score=2.3389
- `schools` (table), score=0.7453
- `schools.CDSCode` (column), score=0.5507
- `frpm.FRPM Count (Ages 5-17)` (column), score=-0.4066
- `frpm` (table), score=-0.5650
- `frpm.Percent (%) Eligible Free (Ages 5-17)` (column), score=-0.6189
- `schools.District` (column), score=-0.6500
- `schools.School` (column), score=-0.7409
- `schools.CharterNum` (column), score=-0.8040
- `schools.AdmEmail2` (column), score=-0.9029
- `satscores` (table), score=-0.9512
- `schools.AdmEmail1` (column), score=-1.0468

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

- `satscores.NumTstTakr` (column), score=2.6617
- `satscores.NumGE1500` (column), score=2.4382
- `schools.Phone` (column), score=2.3396
- `schools` (table), score=0.9695
- `schools.CDSCode` (column), score=0.6323
- `frpm` (table), score=-0.3363
- `schools.District` (column), score=-0.6170
- `schools.School` (column), score=-0.7126
- `satscores` (table), score=-0.7486
- `schools.CharterNum` (column), score=-0.8562
- `schools.AdmEmail2` (column), score=-0.8659
- `schools.AdmEmail1` (column), score=-0.9908
- `schools.EILName` (column), score=-1.1336
- `schools.Ext` (column), score=-1.1828
- `schools.EILCode` (column), score=-1.1828

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

- `schools.School` (column), score=2.9006
- `frpm.Enrollment (Ages 5-17)` (column), score=2.0300
- `schools` (table), score=0.8707
- `schools.CDSCode` (column), score=0.4914
- `frpm` (table), score=-0.4216
- `satscores` (table), score=-0.8414
- `schools.District` (column), score=-1.0854
- `schools.AdmEmail2` (column), score=-1.3323
- `schools.AdmEmail1` (column), score=-1.4423
- `frpm.Charter School Number` (column), score=-1.5367
- `schools.EILName` (column), score=-1.6140
- `schools.Ext` (column), score=-1.6403
- `schools.EILCode` (column), score=-1.6403
- `schools.OpenDate` (column), score=-1.8016
- `schools.Latitude` (column), score=-1.8420

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

- `schools.District` (column), score=2.4081
- `schools.CDSCode` (column), score=0.0086
- `schools` (table), score=-0.0306
- `frpm` (table), score=-0.0898
- `satscores` (table), score=-0.4559
- `frpm.District Name` (column), score=-1.0576
- `satscores.cds` (column), score=-1.1300
- `frpm.District Type` (column), score=-1.1345
- `frpm.District Code` (column), score=-1.3443
- `frpm.School Name` (column), score=-2.0713
- `frpm.Charter School Number` (column), score=-2.1497
- `schools.AdmEmail2` (column), score=-2.1623
- `frpm.School Code` (column), score=-2.1981
- `schools.School` (column), score=-2.2650
- `schools.AdmEmail1` (column), score=-2.2778

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

- `schools` (table), score=1.1766
- `schools.CDSCode` (column), score=0.6328
- `frpm` (table), score=-0.1506
- `satscores` (table), score=-0.5530
- `schools.District` (column), score=-0.8320
- `schools.School` (column), score=-0.9674
- `schools.AdmEmail2` (column), score=-1.0600
- `schools.AdmEmail1` (column), score=-1.1656
- `schools.EILName` (column), score=-1.2885
- `schools.EILCode` (column), score=-1.3384
- `schools.Ext` (column), score=-1.3384
- `schools.OpenDate` (column), score=-1.5961
- `schools.SOC` (column), score=-1.6232
- `schools.Latitude` (column), score=-1.6391
- `schools.AdmFName2` (column), score=-1.6934

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

- `schools.Charter` (column), score=3.0736
- `schools` (table), score=1.2912
- `schools.CDSCode` (column), score=0.6179
- `frpm` (table), score=-0.0263
- `satscores` (table), score=-0.4403
- `frpm.Charter School Number` (column), score=-0.8492
- `schools.District` (column), score=-1.0164
- `schools.CharterNum` (column), score=-1.0746
- `schools.School` (column), score=-1.1129
- `schools.AdmEmail2` (column), score=-1.2220
- `schools.AdmEmail1` (column), score=-1.2762
- `schools.EILName` (column), score=-1.3803
- `schools.EILCode` (column), score=-1.4254
- `schools.Ext` (column), score=-1.4254
- `schools.OpenDate` (column), score=-1.7292

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

- `schools` (table), score=1.0780
- `schools.CDSCode` (column), score=0.8684
- `schools.District` (column), score=0.0009
- `schools.School` (column), score=-0.1489
- `schools.AdmEmail2` (column), score=-0.2457
- `frpm` (table), score=-0.2606
- `schools.AdmEmail1` (column), score=-0.3933
- `schools.EILName` (column), score=-0.5814
- `satscores` (table), score=-0.6215
- `schools.EILCode` (column), score=-0.6395
- `schools.Ext` (column), score=-0.6395
- `schools.OpenDate` (column), score=-0.8106
- `schools.Latitude` (column), score=-0.8437
- `schools.SOC` (column), score=-0.8441
- `schools.AdmFName2` (column), score=-0.9806

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

- `schools.School` (column), score=2.5148
- `schools.Phone` (column), score=0.7763
- `schools` (table), score=0.1681
- `schools.CDSCode` (column), score=0.1664
- `frpm` (table), score=0.1072
- `satscores` (table), score=-0.2529
- `frpm.Charter School Number` (column), score=-0.4781
- `satscores.cds` (column), score=-0.8958
- `frpm.School Name` (column), score=-1.0330
- `frpm.School Type` (column), score=-1.2248
- `frpm.School Code` (column), score=-1.2599
- `frpm.District Name` (column), score=-1.8095
- `schools.District` (column), score=-2.0639
- `schools.AdmEmail2` (column), score=-2.0961
- `frpm.Charter School (Y/N)` (column), score=-2.1186

