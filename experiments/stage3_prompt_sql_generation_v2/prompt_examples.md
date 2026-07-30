# full_schema

DB: `california_schools`

```text
You are an expert SQLite SQL generator.

Rules:
1. Use only the exact table and column names listed in the schema.
2. Do not invent columns or tables.
3. Use SQLite syntax only.
4. Quote column names with backticks if they contain spaces, parentheses, %, /, or hyphens.
5. Return only one SQL query, with no explanation.

Given the database schema and question, generate a valid SQLite SQL query.

Database schema:
Table frpm:
- `2013-14 CALPADS Fall 1 Certification Status` (real)
- `Academic Year` (text)
- `CDSCode` (text)
- `Charter Funding Type` (text)
- `Charter School (Y/N)` (text)
- `Charter School Number` (integer)
- `County Code` (text)
- `County Name` (text)
- `District Code` (text)
- `District Name` (text)
- `District Type` (text)
- `Educational Option Type` (text)
- `Enrollment (Ages 5-17)` (real)
- `Enrollment (K-12)` (text)
- `FRPM Count (Ages 5-17)` (real)
- `FRPM Count (K-12)` (real)
- `Free Meal Count (Ages 5-17)` (real)
- `Free Meal Count (K-12)` (real)
- `High Grade` (text)
- `IRC` (text)
- `Low Grade` (integer)
- `NSLP Provision Status` (text)
- `Percent (%) Eligible FRPM (Ages 5-17)` (real)
- `Percent (%) Eligible FRPM (K-12)` (real)
- `Percent (%) Eligible Free (Ages 5-17)` (real)
- `Percent (%) Eligible Free (K-12)` (real)
- `School Code` (integer)
- `School Name` (text)
- `School Type` (text)

Table satscores:
- `AvgScrMath` (integer)
- `AvgScrRead` (integer)
- `AvgScrWrite` (integer)
- `NumGE1500` (integer)
- `NumTstTakr` (integer)
- `cds` (integer)
- `cname` (text)
- `dname` (text)
- `enroll12` (text)
- `rtype` (text)
- `sname` (text)

Table schools:
- `AdmEmail1` (text)
- `AdmEmail2` (text)
- `AdmEmail3` (text)
- `AdmFName1` (real)
- `AdmFName2` (text)
- `AdmFName3` (text)
- `AdmLName1` (text)
- `AdmLName2` (text)
- `AdmLName3` (text)
- `CDSCode` (integer)
- `Charter` (date)
- `CharterNum` (integer)
- `City` (text)
- `ClosedDate` (date)
- `County` (text)
- `DOC` (text)
- `DOCType` (text)
- `District` (text)
- `EILCode` (text)
- `EILName` (text)
- `EdOpsCode` (text)
- `EdOpsName` (text)
- `Ext` (text)
- `FundingType` (text)
- `GSoffered` (text)
- `GSserved` (text)
- `LastUpdate` (text)
- `Latitude` (integer)
- `Longitude` (real)
- `Magnet` (text)
- `MailCity` (text)
- `MailState` (text)
- `MailStrAbr` (text)
- `MailStreet` (text)
- `MailZip` (text)
- `NCESDist` (text)
- `NCESSchool` (text)
- `OpenDate` (text)
- `Phone` (text)
- `SOC` (text)
- `SOCType` (text)
- `School` (text)
- `State` (text)
- `StatusType` (text)
- `Street` (text)
- `StreetAbr` (text)
- `Virtual` (text)
- `Website` (text)
- `Zip` (text)

Foreign keys:
- frpm.CDSCode = schools.CDSCode
- satscores.cds = schools.CDSCode

Question:
What is the highest eligible free rate for K-12 students in the schools in Alameda County?

Evidence:
Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

Return only the SQL query.
```

# lexical_top30

DB: `california_schools`

```text
You are an expert SQLite SQL generator.

Rules:
1. Use only the exact table and column names listed in the schema.
2. Do not invent columns or tables.
3. Use SQLite syntax only.
4. Quote column names with backticks if they contain spaces, parentheses, %, /, or hyphens.
5. Return only one SQL query, with no explanation.

Given the database schema and question, generate a valid SQLite SQL query.

Database schema:
Table frpm:
- `CDSCode` (text)
- `Enrollment (K-12)` (text)
- `FRPM Count (K-12)` (real)
- `Free Meal Count (K-12)` (real)
- `Percent (%) Eligible FRPM (K-12)` (real)
- `Percent (%) Eligible Free (K-12)` (real)

Table satscores:
- `cds` (integer)

Table schools:
- `AdmEmail1` (text)
- `AdmEmail2` (text)
- `AdmEmail3` (text)
- `AdmFName2` (text)
- `AdmFName3` (text)
- `AdmLName1` (text)
- `AdmLName2` (text)
- `AdmLName3` (text)
- `CDSCode` (integer)
- `CharterNum` (integer)
- `City` (text)
- `County` (text)
- `DOC` (text)
- `DOCType` (text)
- `District` (text)
- `EILCode` (text)
- `EILName` (text)
- `EdOpsCode` (text)
- `EdOpsName` (text)
- `Latitude` (integer)
- `Phone` (text)
- `School` (text)
- `State` (text)
- `Zip` (text)

Foreign keys:
- frpm.CDSCode = schools.CDSCode
- satscores.cds = schools.CDSCode

Question:
What is the highest eligible free rate for K-12 students in the schools in Alameda County?

Evidence:
Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

Return only the SQL query.
```

# rgcn_top30

DB: `california_schools`

```text
You are an expert SQLite SQL generator.

Rules:
1. Use only the exact table and column names listed in the schema.
2. Do not invent columns or tables.
3. Use SQLite syntax only.
4. Quote column names with backticks if they contain spaces, parentheses, %, /, or hyphens.
5. Return only one SQL query, with no explanation.

Given the database schema and question, generate a valid SQLite SQL query.

Database schema:
Table frpm:
- `CDSCode` (text)
- `Enrollment (K-12)` (text)
- `FRPM Count (K-12)` (real)
- `Free Meal Count (K-12)` (real)
- `Percent (%) Eligible FRPM (K-12)` (real)
- `Percent (%) Eligible Free (K-12)` (real)

Table satscores:
- `cds` (integer)

Table schools:
- `AdmEmail1` (text)
- `AdmEmail2` (text)
- `AdmFName3` (text)
- `AdmLName2` (text)
- `AdmLName3` (text)
- `CDSCode` (integer)
- `County` (text)
- `DOC` (text)
- `DOCType` (text)
- `District` (text)
- `EILCode` (text)
- `EILName` (text)
- `Ext` (text)
- `GSserved` (text)
- `Latitude` (integer)
- `NCESDist` (text)
- `OpenDate` (text)
- `SOC` (text)
- `School` (text)
- `State` (text)
- `Virtual` (text)

Foreign keys:
- frpm.CDSCode = schools.CDSCode
- satscores.cds = schools.CDSCode

Question:
What is the highest eligible free rate for K-12 students in the schools in Alameda County?

Evidence:
Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

Return only the SQL query.
```

# rgta_top30

DB: `california_schools`

```text
You are an expert SQLite SQL generator.

Rules:
1. Use only the exact table and column names listed in the schema.
2. Do not invent columns or tables.
3. Use SQLite syntax only.
4. Quote column names with backticks if they contain spaces, parentheses, %, /, or hyphens.
5. Return only one SQL query, with no explanation.

Given the database schema and question, generate a valid SQLite SQL query.

Database schema:
Table frpm:
- `CDSCode` (text)
- `Enrollment (K-12)` (text)
- `FRPM Count (K-12)` (real)
- `Free Meal Count (K-12)` (real)
- `Percent (%) Eligible FRPM (K-12)` (real)
- `Percent (%) Eligible Free (K-12)` (real)

Table satscores:
- `cds` (integer)

Table schools:
- `AdmEmail1` (text)
- `AdmEmail2` (text)
- `AdmFName2` (text)
- `AdmFName3` (text)
- `AdmLName1` (text)
- `AdmLName2` (text)
- `CDSCode` (integer)
- `County` (text)
- `DOC` (text)
- `DOCType` (text)
- `District` (text)
- `EILCode` (text)
- `EILName` (text)
- `Ext` (text)
- `GSserved` (text)
- `Latitude` (integer)
- `NCESDist` (text)
- `OpenDate` (text)
- `SOC` (text)
- `School` (text)
- `State` (text)
- `Virtual` (text)

Foreign keys:
- frpm.CDSCode = schools.CDSCode
- satscores.cds = schools.CDSCode

Question:
What is the highest eligible free rate for K-12 students in the schools in Alameda County?

Evidence:
Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

Return only the SQL query.
```

# oracle_schema

DB: `california_schools`

```text
You are an expert SQLite SQL generator.

Rules:
1. Use only the exact table and column names listed in the schema.
2. Do not invent columns or tables.
3. Use SQLite syntax only.
4. Quote column names with backticks if they contain spaces, parentheses, %, /, or hyphens.
5. Return only one SQL query, with no explanation.

Given the database schema and question, generate a valid SQLite SQL query.

Database schema:
Table frpm:

Question:
What is the highest eligible free rate for K-12 students in the schools in Alameda County?

Evidence:
Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

Return only the SQL query.
```

