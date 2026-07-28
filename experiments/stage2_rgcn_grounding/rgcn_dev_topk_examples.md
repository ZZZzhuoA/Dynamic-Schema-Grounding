## Example 1

**DB:** `california_schools`

**Question:** What is the highest eligible free rate for K-12 students in the schools in Alameda County?

**Evidence:** Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `frpm` (table), score=-0.1810
- `schools` (table), score=-0.6078
- `satscores` (table), score=-0.6421
- `schools.CDSCode` (column), score=-0.8070
- `satscores.cds` (column), score=-1.0289
- `satscores.NumGE1500` (column), score=-1.1849
- `satscores.NumTstTakr` (column), score=-1.1913
- `satscores.rtype` (column), score=-1.2561
- `satscores.cname` (column), score=-1.2984
- `frpm.CDSCode` (column), score=-1.3030
- `satscores.dname` (column), score=-1.3038
- `satscores.AvgScrWrite` (column), score=-1.3870
- `satscores.sname` (column), score=-1.4009
- `satscores.AvgScrRead` (column), score=-1.4035
- `satscores.AvgScrMath` (column), score=-1.4308

## Example 2

**DB:** `california_schools`

**Question:** Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.

**Evidence:** Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `frpm` (table), score=0.0292
- `schools` (table), score=-0.3681
- `satscores` (table), score=-0.4261
- `schools.CDSCode` (column), score=-0.6047
- `satscores.cds` (column), score=-0.8374
- `satscores.NumGE1500` (column), score=-0.9506
- `satscores.NumTstTakr` (column), score=-0.9962
- `satscores.rtype` (column), score=-1.0273
- `satscores.cname` (column), score=-1.0557
- `satscores.dname` (column), score=-1.0673
- `frpm.CDSCode` (column), score=-1.1002
- `satscores.sname` (column), score=-1.1748
- `satscores.AvgScrWrite` (column), score=-1.1852
- `frpm.District Name` (column), score=-1.1871
- `satscores.enroll12` (column), score=-1.2170

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

- `frpm` (table), score=-0.0777
- `satscores` (table), score=-0.3585
- `schools` (table), score=-0.4306
- `schools.CDSCode` (column), score=-0.6558
- `satscores.cds` (column), score=-0.9266
- `satscores.NumGE1500` (column), score=-0.9846
- `satscores.NumTstTakr` (column), score=-1.0564
- `satscores.rtype` (column), score=-1.0612
- `satscores.cname` (column), score=-1.0815
- `satscores.dname` (column), score=-1.1164
- `frpm.CDSCode` (column), score=-1.1791
- `frpm.District Name` (column), score=-1.1888
- `satscores.sname` (column), score=-1.1996
- `satscores.AvgScrWrite` (column), score=-1.2184
- `frpm.Free Meal Count (K-12)` (column), score=-1.2379

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

- `frpm` (table), score=-0.1675
- `schools` (table), score=-0.4257
- `satscores` (table), score=-0.4328
- `schools.CDSCode` (column), score=-0.9046
- `satscores.cds` (column), score=-1.1193
- `satscores.NumGE1500` (column), score=-1.1478
- `satscores.rtype` (column), score=-1.1937
- `satscores.NumTstTakr` (column), score=-1.2064
- `satscores.cname` (column), score=-1.2197
- `satscores.dname` (column), score=-1.2555
- `satscores.AvgScrWrite` (column), score=-1.3836
- `satscores.sname` (column), score=-1.3954
- `satscores.enroll12` (column), score=-1.4038
- `satscores.AvgScrRead` (column), score=-1.4099
- `frpm.District Name` (column), score=-1.4133

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

- `frpm` (table), score=-0.3020
- `satscores` (table), score=-0.6064
- `schools` (table), score=-0.7203
- `schools.CDSCode` (column), score=-0.8983
- `satscores.cds` (column), score=-1.0493
- `satscores.NumGE1500` (column), score=-1.1776
- `satscores.NumTstTakr` (column), score=-1.2402
- `satscores.rtype` (column), score=-1.2578
- `satscores.cname` (column), score=-1.2805
- `frpm.CDSCode` (column), score=-1.3037
- `satscores.dname` (column), score=-1.3075
- `frpm.District Name` (column), score=-1.3386
- `satscores.sname` (column), score=-1.4137
- `satscores.AvgScrWrite` (column), score=-1.4164
- `satscores.AvgScrRead` (column), score=-1.4507

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

- `frpm` (table), score=-0.4101
- `schools` (table), score=-0.7694
- `satscores` (table), score=-0.8226
- `schools.CDSCode` (column), score=-0.9274
- `satscores.cds` (column), score=-1.1254
- `satscores.NumGE1500` (column), score=-1.2532
- `satscores.NumTstTakr` (column), score=-1.3212
- `satscores.rtype` (column), score=-1.3364
- `satscores.cname` (column), score=-1.3522
- `satscores.dname` (column), score=-1.3610
- `frpm.CDSCode` (column), score=-1.3641
- `satscores.sname` (column), score=-1.4801
- `frpm.District Name` (column), score=-1.4825
- `satscores.AvgScrWrite` (column), score=-1.4909
- `satscores.enroll12` (column), score=-1.5129

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

- `frpm` (table), score=-0.0601
- `satscores` (table), score=-0.3610
- `schools` (table), score=-0.4611
- `schools.CDSCode` (column), score=-0.7399
- `satscores.cds` (column), score=-0.8385
- `satscores.NumGE1500` (column), score=-0.9613
- `satscores.NumTstTakr` (column), score=-1.0278
- `satscores.rtype` (column), score=-1.0293
- `satscores.cname` (column), score=-1.0621
- `satscores.dname` (column), score=-1.0699
- `frpm.CDSCode` (column), score=-1.0959
- `satscores.sname` (column), score=-1.1816
- `satscores.AvgScrWrite` (column), score=-1.1954
- `frpm.District Name` (column), score=-1.2222
- `satscores.enroll12` (column), score=-1.2265

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

- `frpm` (table), score=0.1955
- `schools` (table), score=-0.0951
- `satscores` (table), score=-0.1425
- `schools.CDSCode` (column), score=-0.8088
- `satscores.NumGE1500` (column), score=-0.9291
- `satscores.cds` (column), score=-0.9622
- `satscores.NumTstTakr` (column), score=-0.9711
- `satscores.cname` (column), score=-1.0066
- `satscores.rtype` (column), score=-1.0243
- `satscores.dname` (column), score=-1.0442
- `satscores.sname` (column), score=-1.1546
- `satscores.enroll12` (column), score=-1.1741
- `satscores.AvgScrWrite` (column), score=-1.1849
- `satscores.AvgScrRead` (column), score=-1.2395
- `satscores.AvgScrMath` (column), score=-1.2441

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

- `frpm` (table), score=-0.4053
- `satscores` (table), score=-0.7249
- `schools` (table), score=-0.7828
- `schools.CDSCode` (column), score=-1.1531
- `satscores.cds` (column), score=-1.1957
- `satscores.NumGE1500` (column), score=-1.3136
- `satscores.rtype` (column), score=-1.3844
- `satscores.NumTstTakr` (column), score=-1.3867
- `satscores.cname` (column), score=-1.4240
- `satscores.dname` (column), score=-1.4359
- `frpm.CDSCode` (column), score=-1.4913
- `satscores.AvgScrWrite` (column), score=-1.5336
- `satscores.sname` (column), score=-1.5462
- `satscores.enroll12` (column), score=-1.5908
- `satscores.AvgScrRead` (column), score=-1.5930

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

- `frpm` (table), score=-0.5333
- `satscores` (table), score=-0.8588
- `schools` (table), score=-0.9431
- `schools.CDSCode` (column), score=-1.1188
- `satscores.cds` (column), score=-1.2997
- `satscores.NumGE1500` (column), score=-1.4433
- `satscores.NumTstTakr` (column), score=-1.5198
- `satscores.rtype` (column), score=-1.5244
- `frpm.CDSCode` (column), score=-1.5390
- `satscores.cname` (column), score=-1.5547
- `satscores.dname` (column), score=-1.5636
- `frpm.District Name` (column), score=-1.6373
- `satscores.sname` (column), score=-1.6664
- `satscores.AvgScrWrite` (column), score=-1.6707
- `satscores.AvgScrRead` (column), score=-1.7109

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

- `frpm` (table), score=0.0081
- `schools` (table), score=-0.2980
- `satscores` (table), score=-0.3393
- `schools.CDSCode` (column), score=-0.6383
- `satscores.cds` (column), score=-0.9560
- `satscores.NumGE1500` (column), score=-1.0068
- `satscores.NumTstTakr` (column), score=-1.0708
- `satscores.cname` (column), score=-1.0983
- `satscores.rtype` (column), score=-1.1054
- `satscores.dname` (column), score=-1.1386
- `satscores.AvgScrWrite` (column), score=-1.2410
- `satscores.sname` (column), score=-1.2590
- `frpm.CDSCode` (column), score=-1.2696
- `satscores.AvgScrRead` (column), score=-1.2745
- `satscores.enroll12` (column), score=-1.2793

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

- `frpm` (table), score=-0.1537
- `schools` (table), score=-0.5272
- `satscores` (table), score=-0.5960
- `schools.CDSCode` (column), score=-1.0765
- `satscores.cds` (column), score=-1.2107
- `satscores.NumGE1500` (column), score=-1.2963
- `satscores.rtype` (column), score=-1.3592
- `satscores.NumTstTakr` (column), score=-1.3679
- `satscores.dname` (column), score=-1.3971
- `satscores.cname` (column), score=-1.4116
- `frpm.CDSCode` (column), score=-1.4649
- `frpm.District Name` (column), score=-1.5172
- `satscores.sname` (column), score=-1.5222
- `satscores.AvgScrWrite` (column), score=-1.5229
- `frpm.Free Meal Count (K-12)` (column), score=-1.5376

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

- `frpm` (table), score=0.0523
- `schools` (table), score=-0.4066
- `satscores` (table), score=-0.4570
- `schools.CDSCode` (column), score=-0.7745
- `satscores.cds` (column), score=-0.8933
- `satscores.NumGE1500` (column), score=-1.0352
- `satscores.NumTstTakr` (column), score=-1.0713
- `satscores.rtype` (column), score=-1.1093
- `satscores.cname` (column), score=-1.1417
- `satscores.dname` (column), score=-1.1486
- `frpm.CDSCode` (column), score=-1.1833
- `satscores.AvgScrWrite` (column), score=-1.2228
- `satscores.sname` (column), score=-1.2416
- `satscores.AvgScrRead` (column), score=-1.2559
- `satscores.enroll12` (column), score=-1.2843

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

- `frpm` (table), score=-0.3987
- `satscores` (table), score=-0.7689
- `schools` (table), score=-0.8715
- `schools.CDSCode` (column), score=-1.1425
- `satscores.cds` (column), score=-1.1672
- `satscores.NumGE1500` (column), score=-1.2880
- `satscores.NumTstTakr` (column), score=-1.3172
- `satscores.rtype` (column), score=-1.3504
- `satscores.cname` (column), score=-1.3790
- `satscores.dname` (column), score=-1.3904
- `frpm.CDSCode` (column), score=-1.4674
- `satscores.sname` (column), score=-1.4820
- `satscores.AvgScrWrite` (column), score=-1.5021
- `satscores.AvgScrRead` (column), score=-1.5163
- `satscores.enroll12` (column), score=-1.5238

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

- `frpm` (table), score=-0.0030
- `satscores` (table), score=-0.4112
- `schools` (table), score=-0.4212
- `satscores.cds` (column), score=-0.8628
- `schools.CDSCode` (column), score=-0.8925
- `satscores.NumGE1500` (column), score=-0.9225
- `satscores.rtype` (column), score=-0.9228
- `satscores.NumTstTakr` (column), score=-0.9547
- `satscores.cname` (column), score=-0.9573
- `satscores.dname` (column), score=-0.9602
- `satscores.sname` (column), score=-1.0122
- `satscores.AvgScrWrite` (column), score=-1.0779
- `frpm.CDSCode` (column), score=-1.0817
- `satscores.enroll12` (column), score=-1.0864
- `satscores.AvgScrRead` (column), score=-1.0924

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

- `frpm` (table), score=-0.0243
- `satscores` (table), score=-0.4047
- `schools` (table), score=-0.4152
- `schools.CDSCode` (column), score=-0.7005
- `satscores.cds` (column), score=-1.0188
- `satscores.NumGE1500` (column), score=-1.1067
- `satscores.NumTstTakr` (column), score=-1.1688
- `satscores.rtype` (column), score=-1.1926
- `satscores.cname` (column), score=-1.2000
- `satscores.dname` (column), score=-1.2214
- `frpm.CDSCode` (column), score=-1.2758
- `frpm.District Name` (column), score=-1.3300
- `satscores.enroll12` (column), score=-1.3594
- `satscores.sname` (column), score=-1.3601
- `satscores.AvgScrWrite` (column), score=-1.3622

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

- `frpm` (table), score=-0.7862
- `satscores` (table), score=-1.1123
- `schools` (table), score=-1.1984
- `schools.CDSCode` (column), score=-1.5557
- `satscores.cds` (column), score=-1.6013
- `satscores.NumGE1500` (column), score=-1.7102
- `satscores.NumTstTakr` (column), score=-1.7735
- `satscores.rtype` (column), score=-1.7823
- `satscores.cname` (column), score=-1.7992
- `satscores.dname` (column), score=-1.8118
- `frpm.CDSCode` (column), score=-1.8668
- `satscores.sname` (column), score=-1.9373
- `satscores.AvgScrWrite` (column), score=-1.9580
- `satscores.enroll12` (column), score=-1.9723
- `satscores.AvgScrRead` (column), score=-2.0059

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

- `frpm` (table), score=-0.2591
- `satscores` (table), score=-0.5557
- `schools` (table), score=-0.6769
- `satscores.cds` (column), score=-0.8209
- `satscores.NumGE1500` (column), score=-1.0585
- `schools.CDSCode` (column), score=-1.0834
- `satscores.NumTstTakr` (column), score=-1.0918
- `satscores.rtype` (column), score=-1.1359
- `satscores.cname` (column), score=-1.1416
- `satscores.dname` (column), score=-1.1563
- `frpm.CDSCode` (column), score=-1.2222
- `satscores.sname` (column), score=-1.2690
- `satscores.AvgScrWrite` (column), score=-1.2835
- `satscores.AvgScrRead` (column), score=-1.2879
- `satscores.AvgScrMath` (column), score=-1.3084

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

- `frpm` (table), score=-0.6371
- `satscores` (table), score=-0.9414
- `schools` (table), score=-1.0741
- `schools.CDSCode` (column), score=-1.2764
- `satscores.cds` (column), score=-1.3746
- `satscores.NumGE1500` (column), score=-1.5191
- `satscores.NumTstTakr` (column), score=-1.5865
- `satscores.rtype` (column), score=-1.6140
- `frpm.CDSCode` (column), score=-1.6188
- `satscores.cname` (column), score=-1.6251
- `satscores.dname` (column), score=-1.6382
- `frpm.District Name` (column), score=-1.7415
- `satscores.sname` (column), score=-1.7581
- `satscores.AvgScrWrite` (column), score=-1.7749
- `satscores.enroll12` (column), score=-1.7966

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

- `frpm` (table), score=0.0646
- `schools` (table), score=-0.2649
- `satscores` (table), score=-0.2700
- `schools.CDSCode` (column), score=-0.6667
- `satscores.cds` (column), score=-0.9632
- `satscores.NumGE1500` (column), score=-0.9928
- `satscores.NumTstTakr` (column), score=-1.0548
- `satscores.cname` (column), score=-1.0705
- `satscores.rtype` (column), score=-1.0856
- `satscores.dname` (column), score=-1.1191
- `satscores.sname` (column), score=-1.2312
- `satscores.AvgScrWrite` (column), score=-1.2531
- `satscores.enroll12` (column), score=-1.2606
- `frpm.CDSCode` (column), score=-1.2680
- `satscores.AvgScrRead` (column), score=-1.2717

