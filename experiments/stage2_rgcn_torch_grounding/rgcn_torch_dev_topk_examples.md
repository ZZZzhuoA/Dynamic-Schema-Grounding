## Example 1

**DB:** `california_schools`

**Question:** What is the highest eligible free rate for K-12 students in the schools in Alameda County?

**Evidence:** Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `schools` (table), score=2.8046
- `satscores` (table), score=2.4350
- `frpm` (table), score=1.5654
- `satscores.cds` (column), score=0.4268
- `frpm.School Code` (column), score=-0.0826
- `satscores.NumGE1500` (column), score=-0.1216
- `schools.CDSCode` (column), score=-0.1390
- `frpm.District Name` (column), score=-0.1721
- `schools.SOC` (column), score=-0.1847
- `frpm.Charter School Number` (column), score=-0.2071
- `frpm.Charter School (Y/N)` (column), score=-0.2221
- `schools.District` (column), score=-0.2489
- `schools.AdmEmail2` (column), score=-0.2541
- `schools.School` (column), score=-0.2557
- `schools.OpenDate` (column), score=-0.2802

## Example 2

**DB:** `california_schools`

**Question:** Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.

**Evidence:** Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `schools` (table), score=2.7480
- `satscores` (table), score=1.6420
- `frpm` (table), score=0.4851
- `satscores.cds` (column), score=-0.1346
- `schools.SOC` (column), score=-0.1952
- `frpm.School Code` (column), score=-0.5471
- `satscores.AvgScrWrite` (column), score=-0.6432
- `frpm.Low Grade` (column), score=-0.6830
- `satscores.NumGE1500` (column), score=-0.6875
- `schools.DOC` (column), score=-0.7070
- `frpm.Charter School Number` (column), score=-0.7732
- `schools.AdmFName1` (column), score=-0.7767
- `schools.Longitude` (column), score=-0.8411
- `frpm.Charter School (Y/N)` (column), score=-0.8698
- `frpm.2013-14 CALPADS Fall 1 Certification Status` (column), score=-0.8880

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

- `frpm` (table), score=3.2964
- `schools` (table), score=2.8320
- `satscores` (table), score=2.6431
- `frpm.District Name` (column), score=0.8601
- `frpm.District Code` (column), score=0.4994
- `schools.District` (column), score=0.3706
- `frpm.School Name` (column), score=0.3630
- `frpm.Charter School Number` (column), score=0.3426
- `schools.AdmEmail2` (column), score=0.3376
- `frpm.School Code` (column), score=0.3104
- `frpm.District Type` (column), score=0.2761
- `frpm.Low Grade` (column), score=0.2164
- `schools.CDSCode` (column), score=0.2097
- `frpm.Free Meal Count (K-12)` (column), score=0.1839
- `frpm.FRPM Count (Ages 5-17)` (column), score=0.1789

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

- `frpm` (table), score=3.8953
- `satscores` (table), score=3.0555
- `schools` (table), score=2.7203
- `frpm.District Name` (column), score=1.0383
- `frpm.CDSCode` (column), score=0.6229
- `frpm.District Code` (column), score=0.5901
- `frpm.County Name` (column), score=0.5329
- `frpm.District Type` (column), score=0.3326
- `frpm.School Name` (column), score=0.1846
- `schools.CDSCode` (column), score=0.0806
- `schools.NCESSchool` (column), score=0.0143
- `schools.GSoffered` (column), score=0.0143
- `schools.AdmEmail1` (column), score=0.0045
- `schools.District` (column), score=-0.0586
- `schools.AdmEmail2` (column), score=-0.0617

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

- `schools` (table), score=3.4615
- `satscores` (table), score=3.4412
- `frpm` (table), score=3.1968
- `frpm.District Name` (column), score=0.9969
- `frpm.County Name` (column), score=0.7825
- `satscores.cds` (column), score=0.5342
- `schools.CDSCode` (column), score=0.4968
- `frpm.School Name` (column), score=0.3748
- `frpm.District Code` (column), score=0.2880
- `schools.NCESSchool` (column), score=0.2250
- `schools.GSoffered` (column), score=0.2250
- `frpm.District Type` (column), score=0.1591
- `schools.EILCode` (column), score=0.1353
- `schools.Ext` (column), score=0.1353
- `schools.District` (column), score=0.0945

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

- `frpm` (table), score=5.6022
- `schools` (table), score=4.6174
- `satscores` (table), score=4.2151
- `frpm.District Name` (column), score=2.0904
- `frpm.County Name` (column), score=1.3154
- `frpm.District Code` (column), score=1.0774
- `frpm.School Name` (column), score=0.8203
- `schools.CDSCode` (column), score=0.6435
- `frpm.District Type` (column), score=0.5493
- `frpm.CDSCode` (column), score=0.5393
- `schools.District` (column), score=0.2957
- `frpm.Charter School Number` (column), score=0.2162
- `schools.NCESSchool` (column), score=0.1985
- `schools.GSoffered` (column), score=0.1985
- `schools.AdmEmail2` (column), score=0.1743

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

- `frpm` (table), score=3.7074
- `schools` (table), score=3.3978
- `satscores` (table), score=3.2272
- `frpm.District Name` (column), score=0.8462
- `frpm.County Name` (column), score=0.2363
- `frpm.District Code` (column), score=0.1909
- `schools.CDSCode` (column), score=0.0672
- `frpm.District Type` (column), score=-0.0276
- `frpm.School Name` (column), score=-0.0406
- `frpm.Charter School Number` (column), score=-0.1986
- `schools.District` (column), score=-0.2230
- `schools.AdmEmail2` (column), score=-0.2467
- `satscores.cds` (column), score=-0.2487
- `frpm.School Code` (column), score=-0.2634
- `schools.GSoffered` (column), score=-0.3008

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

- `schools` (table), score=3.3741
- `satscores` (table), score=3.0062
- `frpm` (table), score=2.8762
- `frpm.District Name` (column), score=0.3329
- `satscores.cds` (column), score=0.0688
- `schools.CDSCode` (column), score=-0.0330
- `frpm.County Name` (column), score=-0.0559
- `schools.NCESSchool` (column), score=-0.4002
- `schools.GSoffered` (column), score=-0.4002
- `schools.Ext` (column), score=-0.4077
- `schools.EILCode` (column), score=-0.4077
- `frpm.District Code` (column), score=-0.4305
- `frpm.School Name` (column), score=-0.4477
- `schools.Latitude` (column), score=-0.5137
- `frpm.Charter School Number` (column), score=-0.5367

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

- `frpm` (table), score=3.9773
- `schools` (table), score=3.2460
- `satscores` (table), score=3.1746
- `frpm.District Name` (column), score=1.1114
- `frpm.County Name` (column), score=0.6001
- `frpm.District Code` (column), score=0.4552
- `frpm.CDSCode` (column), score=0.2434
- `schools.CDSCode` (column), score=0.2115
- `frpm.District Type` (column), score=0.1905
- `frpm.School Name` (column), score=0.1231
- `schools.NCESSchool` (column), score=0.0442
- `schools.GSoffered` (column), score=0.0442
- `schools.District` (column), score=-0.1203
- `schools.AdmEmail2` (column), score=-0.1432
- `schools.AdmEmail1` (column), score=-0.1561

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

- `satscores` (table), score=3.3112
- `frpm` (table), score=3.2623
- `schools` (table), score=3.2145
- `frpm.District Name` (column), score=0.9375
- `frpm.County Name` (column), score=0.7911
- `schools.AdmEmail1` (column), score=0.4062
- `schools.CDSCode` (column), score=0.3508
- `frpm.CDSCode` (column), score=0.3062
- `frpm.District Code` (column), score=0.2574
- `schools.NCESSchool` (column), score=0.2258
- `schools.GSoffered` (column), score=0.2258
- `satscores.cds` (column), score=0.2095
- `frpm.School Name` (column), score=0.1361
- `frpm.District Type` (column), score=0.1226
- `schools.Ext` (column), score=0.0903

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

- `schools` (table), score=3.6605
- `satscores` (table), score=3.1748
- `frpm` (table), score=2.1233
- `satscores.cds` (column), score=0.8530
- `frpm.District Name` (column), score=0.3915
- `schools.CDSCode` (column), score=0.3216
- `frpm.County Name` (column), score=0.2243
- `schools.SOC` (column), score=0.1471
- `schools.EILCode` (column), score=0.1212
- `schools.Ext` (column), score=0.1212
- `frpm.School Code` (column), score=-0.0594
- `frpm.School Name` (column), score=-0.0633
- `schools.Charter` (column), score=-0.0712
- `schools.ClosedDate` (column), score=-0.1013
- `schools.Latitude` (column), score=-0.1215

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

- `frpm` (table), score=6.4808
- `schools` (table), score=5.1220
- `satscores` (table), score=4.3750
- `frpm.District Name` (column), score=2.5571
- `frpm.District Code` (column), score=1.3826
- `frpm.County Name` (column), score=1.3645
- `frpm.School Name` (column), score=1.0330
- `schools.CDSCode` (column), score=0.8246
- `frpm.District Type` (column), score=0.7362
- `frpm.Charter School Number` (column), score=0.5118
- `schools.District` (column), score=0.5118
- `frpm.CDSCode` (column), score=0.4799
- `schools.AdmEmail2` (column), score=0.3529
- `schools.NCESSchool` (column), score=0.1339
- `schools.GSoffered` (column), score=0.1339

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

- `schools` (table), score=1.6937
- `satscores` (table), score=0.6178
- `satscores.AvgScrWrite` (column), score=-0.4032
- `schools.SOC` (column), score=-0.4250
- `satscores.cds` (column), score=-0.6453
- `schools.DOC` (column), score=-0.7746
- `schools.Longitude` (column), score=-0.9006
- `schools.AdmFName1` (column), score=-0.9412
- `satscores.NumGE1500` (column), score=-0.9751
- `satscores.NumTstTakr` (column), score=-1.0837
- `frpm.School Code` (column), score=-1.0864
- `frpm` (table), score=-1.1006
- `schools.CharterNum` (column), score=-1.1564
- `schools.School` (column), score=-1.1634
- `frpm.Percent (%) Eligible FRPM (K-12)` (column), score=-1.1881

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

- `satscores` (table), score=1.7695
- `schools` (table), score=1.7499
- `frpm` (table), score=0.4702
- `satscores.cds` (column), score=-0.2317
- `satscores.enroll12` (column), score=-0.7564
- `schools.CDSCode` (column), score=-0.7599
- `schools.GSserved` (column), score=-0.8334
- `satscores.NumGE1500` (column), score=-0.8345
- `schools.Longitude` (column), score=-0.8487
- `satscores.rtype` (column), score=-0.8649
- `satscores.NumTstTakr` (column), score=-0.8685
- `satscores.AvgScrWrite` (column), score=-0.9307
- `schools.AdmLName2` (column), score=-0.9477
- `frpm.County Name` (column), score=-0.9542
- `schools.CharterNum` (column), score=-0.9698

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

- `frpm` (table), score=4.0969
- `schools` (table), score=2.3373
- `satscores` (table), score=1.6489
- `frpm.District Name` (column), score=0.7271
- `frpm.District Code` (column), score=0.5039
- `frpm.2013-14 CALPADS Fall 1 Certification Status` (column), score=0.3253
- `frpm.Charter School Number` (column), score=0.2899
- `frpm.School Code` (column), score=0.2242
- `schools.District` (column), score=0.1888
- `frpm.Low Grade` (column), score=0.1636
- `frpm.District Type` (column), score=0.1254
- `schools.AdmEmail2` (column), score=0.1071
- `frpm.FRPM Count (K-12)` (column), score=-0.1073
- `frpm.School Name` (column), score=-0.1151
- `frpm.Charter School (Y/N)` (column), score=-0.1198

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

- `frpm` (table), score=4.5986
- `schools` (table), score=4.2906
- `satscores` (table), score=4.1699
- `frpm.District Name` (column), score=1.6812
- `frpm.County Name` (column), score=1.1862
- `frpm.District Code` (column), score=0.8237
- `frpm.School Name` (column), score=0.7821
- `schools.CDSCode` (column), score=0.7089
- `satscores.cds` (column), score=0.5225
- `frpm.CDSCode` (column), score=0.4512
- `frpm.District Type` (column), score=0.4286
- `frpm.Charter School Number` (column), score=0.3593
- `schools.District` (column), score=0.3340
- `schools.NCESSchool` (column), score=0.2973
- `schools.GSoffered` (column), score=0.2973

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

- `frpm` (table), score=5.8862
- `schools` (table), score=2.3515
- `satscores` (table), score=2.2272
- `frpm.District Name` (column), score=1.3636
- `frpm.District Code` (column), score=0.8770
- `frpm.CDSCode` (column), score=0.1001
- `frpm.District Type` (column), score=0.0852
- `frpm.County Name` (column), score=-0.5223
- `schools.CDSCode` (column), score=-0.6196
- `schools.District` (column), score=-0.6277
- `frpm.Enrollment (K-12)` (column), score=-0.6657
- `frpm.Charter School Number` (column), score=-0.6707
- `frpm.Low Grade` (column), score=-0.7091
- `schools.AdmEmail2` (column), score=-0.7348
- `frpm.School Name` (column), score=-0.7511

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

- `satscores` (table), score=4.0874
- `frpm` (table), score=3.5736
- `schools` (table), score=3.3573
- `schools.AdmEmail1` (column), score=1.7570
- `frpm.County Name` (column), score=1.6255
- `frpm.CDSCode` (column), score=1.5565
- `frpm.District Name` (column), score=1.2860
- `schools.SOCType` (column), score=1.1016
- `schools.NCESSchool` (column), score=0.6274
- `schools.GSoffered` (column), score=0.6274
- `frpm.NSLP Provision Status` (column), score=0.6039
- `satscores.sname` (column), score=0.4864
- `schools.AdmLName2` (column), score=0.4649
- `schools.CDSCode` (column), score=0.4404
- `satscores.dname` (column), score=0.4036

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

- `frpm` (table), score=5.6540
- `satscores` (table), score=3.5005
- `schools` (table), score=3.2239
- `frpm.District Name` (column), score=1.9328
- `frpm.CDSCode` (column), score=1.2743
- `frpm.District Code` (column), score=1.1321
- `frpm.County Name` (column), score=1.0399
- `frpm.District Type` (column), score=0.5238
- `frpm.School Name` (column), score=0.1122
- `schools.CDSCode` (column), score=0.1050
- `frpm.NSLP Provision Status` (column), score=-0.0287
- `schools.NCESSchool` (column), score=-0.0404
- `schools.GSoffered` (column), score=-0.0404
- `frpm.Enrollment (K-12)` (column), score=-0.0720
- `schools.AdmEmail1` (column), score=-0.0869

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

- `schools` (table), score=3.3449
- `satscores` (table), score=3.0325
- `frpm` (table), score=2.6414
- `frpm.District Name` (column), score=0.4367
- `satscores.cds` (column), score=0.3714
- `schools.CDSCode` (column), score=0.0934
- `frpm.School Code` (column), score=0.0097
- `frpm.County Name` (column), score=-0.0058
- `frpm.Charter School Number` (column), score=-0.0405
- `frpm.School Name` (column), score=-0.0564
- `schools.District` (column), score=-0.0716
- `schools.AdmEmail2` (column), score=-0.0997
- `frpm.Charter School (Y/N)` (column), score=-0.1648
- `frpm.District Code` (column), score=-0.2093
- `satscores.NumGE1500` (column), score=-0.2182

