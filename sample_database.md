/*Table structure for table `final_balance_sheet_new` */

DROP TABLE IF EXISTS `final_balance_sheet_new`;

CREATE TABLE `final_balance_sheet_new` (
  `id` int NOT NULL AUTO_INCREMENT,
  `account_type` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `SQL_Account_Name_Code` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `SQL_Account_Name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `SQL_Account_Category_Order_Code` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `SQL_Account_Category_Order` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `Sub_Account_Category_Order_Code` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `Sub_Account_Category_Order` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `SQL_Account_Group_Name_Code` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `SQL_Account_Group_Name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `SQL_Sub_Account_Group_Name_Code` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `SQL_Sub_Account_Group_Name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `DC_BS_Account_Name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `Amount` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `Total_Amount` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `Prior` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `Operator` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `SQL_BS_Account_ID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `SQL_Property` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `Month` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `account_type` (`account_type`,`SQL_Account_Name_Code`,`SQL_Account_Name`,`SQL_Account_Category_Order_Code`,`SQL_Account_Category_Order`,`Sub_Account_Category_Order_Code`),
  KEY `Sub_Account_Category_Order` (`Sub_Account_Category_Order`,`SQL_Account_Group_Name_Code`,`SQL_Account_Group_Name`,`SQL_Sub_Account_Group_Name_Code`,`SQL_Sub_Account_Group_Name`,`DC_BS_Account_Name`),
  KEY `Prior` (`Prior`,`Operator`,`SQL_Property`,`Month`),
  KEY `SQL_BS_Account_ID` (`SQL_BS_Account_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=17796 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;