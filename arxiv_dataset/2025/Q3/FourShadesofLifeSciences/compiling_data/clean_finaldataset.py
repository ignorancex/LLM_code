#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "clean raw data for FSoLS data set"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2024-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "

import pandas as pd
import re
import argparse


def clean_pmc(base_url, input):
    df = pd.read_csv(input)

    df["data-source"] = "PMC"
    print(df.head())

    patterns = [
        r" Open Access This article is licensed under a Creative Commons Attribution 4.0 International License which permits use sharing adaptation distribution and reproduction in any medium or format as long as you give appropriate credit to the original authors and the source provide a link to the Creative Commons licence and indicate if changes were made. The images or other third party material in this article are included in the articles Creative Commons licence unless indicated otherwise in a credit line to the material. If material is not included in the articles Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use you will need to obtain permission directly from the copyright holder. To view a copy of this licence visit httpcreativecommons.orglicensesby4.0. The Creative Commons Public Domain Dedication waiver httpcreativecommons.orgpublicdomainzero1.0 applies to the data made available in this article unless otherwise stated in a credit line to the data.",
        r"This is an open access article distributed under the terms of the Creative Commons Attribution (?:NonCommercial|License|CC BY(?:-NC|-ND|-NC-ND|-SA)?)",
        r"This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution",
        r"This is an open access article under the CC BYNCND license",
        r"This is an open access article under the terms of the https?://creativecommons\.org/licenses/by-nc-nd/4\.0 License which permits use and distribution in any medium provided the original work is properly cited, the use is noncommercial and no modifications or adaptations are made\.",
        r"This is an open access article under the CC BY license https?://creativecommons\.org/licenses/by/4\.0",
        r"Open Access This article is <[a-zA-Z]+\.[a-zA-Z]+\.[a-zA-Z]+\.> https?://creativecommons\.org/licenses/(by(?:-nc-nd)?[2-4]\.0|publicdomainzero1\.0)",
        r"https?://creativecommons\.org/licenses/(by(?:-nc-nd)?[2-4]\.0|publicdomainzero1\.0)",
        r"This work is published and licensed by Dove Medical Press Limited\.",
        r"The full terms of this license are available at https://www\.dovepress\.com/terms\.php and incorporate the Creative Commons Attribution Non Commercial unported v3\.0 License https?://creativecommons\.org/licenses/by-nc/3\.0\. By accessing the work you hereby accept the Terms\. Noncommercial uses of the work are permitted without any further permission from Dove Medical Press Limited provided the work is properly attributed\.",
        r"Open Access This article is licensed under a Creative Commons Attribution 4\.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author\(s\), and the source, provide a link to the Creative Commons licence, and indicate if changes were made\. The images or other third party material in this article are included in the article\'s Creative Commons licence, unless indicated otherwise in a credit line to the material\. If material is not included in the article\'s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder\. To view a copy of this licence, visit http://creativecommons\.org/licenses/by/4\.0\. The Creative Commons Public Domain Dedication waiver http://creativecommons\.org/publicdomain/zero/1\.0 applies to the data made available in this article, unless otherwise stated in a credit line to the data\.",
        r"Reuse permitted under CC BYNC\. No commercial reuse\. See rights and permissions\. Published by BMJ\. 2021 https?://creativecommons\.org/licenses/by-nc/4\.0 This is an open access article distributed in accordance with the Creative Commons Attribution Non Commercial CC BYNC 4\.0 license which permits others to distribute, remix, adapt, build upon this work noncommercially, and license their derivative works on different terms, provided the original work is properly cited, appropriate credit is given, any changes made indicated, and the use is noncommercial\. See http://creativecommons\.org/licenses/by-nc/4\.0\.",
        r"This is an open access article distributed under the Creative Commons Attribution License 4\.0 CCBY",
        r"nSubmit your next manuscript to BioMed Central and take full advantage of n Convenient online submission n Thorough peer review n No space constraints or color figure charges n Immediate publication on acceptance n Inclusion in PubMed, CAS, Scopus and Google Scholar n Research which is freely available for redistribution nSubmit your manuscript at submit",
        r"This work is published by Dove Medical Press Limited and licensed under Creative Commons Attribution Non Commercial unported v3\.0 License 2015 The full terms of the License are available at https?://creativecommons\.org/licenses/by-nc/3\.0\. Noncommercial uses of the work are permitted without any further permission from Dove Medical Press Limited provided the work is properly attributed\.",
        r"This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution CC BY license",
        r"This article is licensed under a Creative Commons Attribution 4\.0 International License which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original authors and the source, provide a link to the Creative Commons licence, and indicate if changes were made\. The images or other third party material in this article are included in the article\'s Creative Commons licence, unless indicated otherwise in a credit line to the material\. If material is not included in the article\'s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder\. To view a copy of this licence, visit \.",
        r"Open Access The Creative Commons Public Domain Dedication waiver applies to the data made available in this article unless otherwise stated in a credit line to the data\.",
        r"issuecopyrightstatement",
        r"This is an open access article distributed under the terms of the Creative Commons Attribution License which permits unrestricted use, distribution and reproduction in any medium provided the original author and source are credited\.",
        r"This is an openaccess article distributed under the terms of the Creative Commons Attribution 4\.0 International license which permits unrestricted use, distribution and reproduction in any medium provided that the original work is properly attributed\.",
        r"This is an openaccess article distributed under the terms of the Creative Commons Attribution License which permits unrestricted use, distribution and reproduction in any medium provided the original author and source are properly credited\.",
        r"Open Access The Creative Commons Public Domain Dedication waiver applies to the data made available in this article unless otherwise stated in a credit line to the data\.",
        r"This is an open access article distributed under the terms of the Creative Commons AttributionNon Commercial License 4\.0 CCBYNC where it is permissible to download, share, remix, transform and buildup the work provided it is properly cited\. The work cannot be used commercially without permission from the journal\.",
        r"This is an open access article distributed under the Creative Commons Attribution License which permits unrestricted use, distribution and reproduction in any medium provided the original work is properly cited\.",
        r"This article is distributed under the terms of the Creative CommonsAttributionNonCommercial 4\.0 License https?://creativecommons\.org/licenses/by-nc/4\.0 which permits noncommercial use, reproduction and distribution of the work without further permission provided the original work is attributed as specified on the SAGE and Open Access page https://us\.sagepub\.com/en-us/nam/open-access-at-sage\.",
        r"This is an openaccess article distributed under the terms of the Creative Commons License",
        r"This is an open access article distributed in accordance with the Creative Commons Attribution Non Commercial CC BYNC 4\.0 license which permits others to distribute, remix, adapt, build upon this work noncommercially, and license their derivative works on different terms, provided the original work is properly cited, appropriate credit is given, any changes made indicated, and the use is noncommercial\.",
        r"https?://creativecommons\.org/licenses/by-nc/4\.0",
        r"Supplementary Information The online version contains supplementary material available at",
        r"Keywords",
        r"Open Access The Creative Commons Public Domain Dedication waiver applies to the data made available in this article unless otherwise stated in a credit line to the data\. Purpose",
        r"This is an Open Access article distributed under the terms of the Creative Commons Attribution NonCommercial License which permits noncommercial reuse, distribution and reproduction in any medium provided the original work is properly cited\.",
        r"BioMed Central",
        r"Correspondence",
        r"Licensee MDPI, Basel, Switzerland\.",
        r"licensee BioMed Central Ltd\.",
        r"Data curation",
        r"OPENACCESS",
        r"Published by Elsevier Inc\.",
        r"Elsevier Ltd\.",
        r"This article is copyright of the authors or their affiliated institutions",
        r"Open Access This article is licensed under a Creative Commons Attribution 4\.0 International License which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original authors and the source, provide a link to the Creative Commons license, and indicate if changes were made\. The images or other third party material in this article are included in the article\'s Creative Commons license, unless indicated otherwise in a credit line to the material\. If material is not included in the article\'s Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder\. To view a copy of this license, visit",
        r"This is an open access article distributed under the terms of the Creative Commons Attribution NonCommercial License https?://creativecommons\.org/licenses/by-nc/4\.0 which permits unrestricted non-commercial use, distribution and reproduction in any medium provided the original author and source are credited\.",
        r"John Wiley & Sons Australia Ltd",
        r"Department, Daiichi Sankyo Co\. Ltd\., Tokyo, Japan",
        r"Blackwell Publishing Ltd",
        r"BMJ Publishing Group Ltd\.",
        r"Meyer et al\. licensee Ltd",
        r"Editora Ltda\.",
        r"Elsevier",
        r"Link\.Springer\.com",
        r"The Authors under exclusive licence to Springer Medizin Verlag GmbH, ein Teil von Springer Nature",
        r"Springer International Publishing",
        r"Springer Nature Limited",
        r"This is an open access article distributed under the terms of the Creative Commons Attribution NonCommercial CC BYNC 4\.0 License which permits unrestricted noncommercial use, distribution and reproduction in any medium provided the original work is properly cited\.",
        r"This article is distributed under the terms of the Creative Commons Attribution 4\.0 International License which permits unrestricted use, distribution and reproduction in any medium provided you give appropriate credit to the original authors and the source, provide a link to the Creative Commons license, and indicate if changes were made\. The Creative Commons Public Domain Dedication waiver applies to the data made available in this article unless otherwise stated\.",
        r"The Creative Commons Public Domain Dedication waiver applies to the data made available in this article unless otherwise stated\.",
        r"Copyright \(199[0-9]|20[01][0-9]|202[0-5]\)",
        r"This article is distributed under the terms of the Creative Commons Attribution 4\.0 License which permits any use, reproduction and distribution of the work without further permission provided the original work is attributed as specified on the SAGE and Open Access pages https://us\.sagepub\.com/en-us/nam/open-access-at-sage\.",
        r"This is an Open Access article distributed under the terms of the Creative Commons Attribution NonCommercial License https?://creativecommons\.org/licenses/by-nc/4\.0 which permits noncommercial reuse, distribution and reproduction in any medium provided the original work is properly cited\. For commercial reuse, please contact journals\.permissions@oup\.com",
        r"This is an Open Access article distributed under the terms of the Creative Commons Attribution NonCommercial License which permits unrestricted noncommercial use, distribution and reproduction in any medium provided the original work is properly cited\.",
        r"This is an open access article published by Thieme under the terms of the Creative Commons Attribution 4\.0 International License permitting copying and reproduction so long as the original work is given appropriate credit",
        r"This is an openaccess article distributed under the terms of the Creative Commons Attribution License which permits unrestricted use, distribution and reproduction in any medium provided the original work is properly cited\.",
        r"This is an open access article distributed under the terms of the Creative Commons Attribution License CC BY 3\.0 which permits unrestricted use, distribution and reproduction in any medium provided the original author and source are credited\.",
        r"coverdate\s+[A-Za-z]+\s+\d{4}\s+detailsofpublishersconvertor\s+ConverterWILEYML3GV2TOJATSPMC\s+version\d+\.\d+\.\d+\s+moderemoveFC\s+converted\d{2}\.\d{2}\.\d{4}",
        r"This work is licensed under the Creative Commons Attribution License\.",
        r"This is an openaccess article distributed under the terms of the Creative Commons Attribution License CC BY\. The use, distribution or reproduction in other forums is permitted provided the original authors and the copyright owners are credited and that the original publication in this journal is cited in accordance with accepted academic practice\. No use, distribution or reproduction is permitted which does not comply with these terms\.",
        r"which permits unrestricted reuse, distribution and reproduction in any medium provided the original work is properly cited\.",
        r"This article is made available via the PMC Open Access Subset for unrestricted reuse and analyses in any form or by any means with acknowledgement of the original source\. These permissions are granted for the duration of the COVID19 pandemic or until permissions are revoked in writing\. Upon expiration of these permissions, PMC is granted a perpetual license to make this article available via PMC and Europe PMC consistent with existing copyright protections\.",
        r"This is an openaccess article distributed under the terms of the Creative Commons Attribution License CC BY\. The use, distribution or reproduction in other forums is permitted provided the original authors and the copyright owners are credited and that the original publication in this journal is cited in accordance with accepted academic practice\. No use, distribution or reproduction is permitted which does not comply with these terms\.",
        r"This work is licensed under Creative Common AttributionNonCommercialNoDerivatives 4\.0 International CC BYNCND 4\.0",
        r"This is an openaccess article distributed under the terms of the Creative Commons Attribution License",
        r"This work is licensed under the Creative Commons AttributionNonCommercialNo Derivative Works 3\.0 Unported License\. To view a copy of this license, visit",
        r"Manuscript content on this site is licensed under Creative Commons Licenses",
        r"For commercial reuse, please contact journals\.permissions@oup\.com",
        r"This is an Open Access article distributed under the terms of the Creative Commons AttributionNonCommercialNoDerivs licence which permits noncommercial reproduction and distribution of the work in any medium provided the original work is not altered or transformed in any way and that the work is properly cited\.",
        r"Since January 2020, has created a COVID19 resource centre with free information in English and Mandarin on the novel coronavirus COVID19\. The COVID19 resource centre is hosted on Connect, the company\'s public news and information website\. hereby grants permission to make all its COVID19related research that is available on the COVID19 resource centre, including this research content, immediately available in PubMed Central and other publicly funded repositories such as the WHO COVID database with rights for unrestricted research, reuse and analyses in any form or by any means with acknowledgement of the original source\. These permissions are granted for free by for as long as the COVID19 resource centre remains active\.",
        r"Manuscript content on this site is licensed under Creative Commons Licenses",
        r"http://creativecommons\.org/licenses/by-nc/4\.0",
        r"This article is Open Access CC BY 4\.0 licence http://creativecommons\.org/licenses/by/4\.0\.",
        r"This work is licensed under a Creative Commons AttributionNonCommercial 4\.0 International License",
        r"This is an openaccess article distributed under the terms of the Creative Commons AttributionNon CommercialNo Derivatives License 4\.0 CCBYNCND where it is permissible to download and share the work provided it is properly cited\. The work cannot be changed in any way or used commercially without permission from the journal\. Supplemental digital content is available in the text\.",
        r"This is an openaccess article distributed under the terms of the Creative Commons AttributionNon Commercial License 4\.0 CCBYNC where it is permissible to download, share, remix, transform and buildup the work provided it is properly cited\. The work cannot be used commercially without permission from the journal\.",
        r"\d+department",
        r"License which permits use distribution and reproduction in any medium provided the original work is properly cited and is not used for commercial purposes.",
        r"This is an Open Access article distributed under the terms of the Creative Commons AttributionNonCommercialNoDerivs licence  which permits noncommercial reproduction and distribution of the work in any medium provided the original work is not altered or transformed in any way and that the work is properly cited.",
        r"https?://creativecommons\.org/licenses/(by|by-nc-nd)-?(\d\.\d)",
        r"Open AccessThis article is licensed under a Creative Commons Attribution 4.0 International License which permits use sharing adaptation distribution and reproduction in any medium or format as long as you give appropriate credit to the original authors and the source provide a link to the Creative Commons licence and indicate if changes were made. The images or other third party material in this article are included in the articles Creative Commons licence unless indicated otherwise in a credit line to the material. If material is not included in the articles Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use you will need to obtain permission directly from the copyright holder. To view a copy of this licence visit httpcreativecommons.orglicensesby4.0. The Creative Commons Public Domain Dedication waiver httpcreativecommons.orgpublicdomainzero1.0 applies to the data made available in this article unless otherwise stated in a credit line to the data.",
        r"This is an open access article distributed in accordance with the Creative Commons Attribution 4.0 Unported CC BY 4.0 license which permits others to copy redistribute remix transform and build upon this work for any purpose provided the original work is properly cited a link to the licence is given and indication of whether changes were made.",
        r"sourceschemaversionnumber",
        r"\b[1-9]Department\b",
        r"https?creativecommons\.org\/licenses\/[a-z]+(?:[a-z])?[0-9]\.[0-9]",
        r"The use distribution or reproduction in other forums is permitted provided the original authors and the copyright owners are credited and that the original publication in this journal is cited in accordance with accepted academic practice. No use distribution or reproduction is permitted which does not comply with these terms.",
        r"The full terms of this license are available at httpswww.dovepress.comterms.php and incorporate the Creative Commons Attribution  Non Commercial unported v3.0 License",
        r"Licensee MDPI Basel Switzerland",
        r"licensee  Ltd.",
        r"This is an open access article distributed under the Creative Commons Attribution License which permits unrestricted use distribution and reproduction in any medium provided the original work is properly cited.",
        r"This is an openaccess article distributed under the terms of the        Creative Commons Attribution License which permits unrestricted use        distribution and reproduction in any medium provided the original        work is properly cited.",
        r"This is an open access article under the terms of the Creative Commons Attribution License which permits use distribution and reproduction in any medium provided that the original work is properly cited.",
        r"This is an open access article distributed under the terms of the Creative Commons license.",
        r"This is an open access article distributed in accordance with the Creative Commons Attribution Non Commercial CC BYNC 4.0 license which permits others to distribute remix adapt build upon this work noncommercially and license their derivative works on different terms provided the original work is properly cited appropriate credit is given any changes made indicated and the use is noncommercial. Seehttpcreativecommons.orglicensesbync4.0.",
        r"Open Access This article is distributed under the terms of the Creative Commons Attribution 4.0 International License httpcreativecommons.orglicensesby4.0 which permits unrestricted use distribution and reproduction in any medium provided you give appropriate credit to the original authors and the source provide a link to the Creative Commons license and indicate if changes were made. The Creative Commons Public Domain Dedication waiver httpcreativecommons.orgpublicdomainzero1.0 applies to the data made available in this article unless otherwise stated.",
        r"This is an Open Access article distributed under the terms of the Creative Commons Attribution License httpcreativecommons.orglicensesby4.0 which permits unrestricted use distribution and reproduction in any medium provided the original work is properly credited. The Creative Commons Public Domain Dedication waiver httpcreativecommons.orgpublicdomainzero1.0 applies to the data made available in this article unless otherwise stated.",
        r"Open Access This article is licensed under a Creative Commons Attribution 4.0 International License which permits use sharing adaptation distribution and reproduction in any medium or format as long as you give appropriate credit to the original authors and the source provide a link to the Creative Commons license and indicate if changes were made. The images or other third party material in this article are included in the articles Creative Commons license unless indicated otherwise in a credit line to the material. If material is not included in the articles Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use you will need to obtain permission directly from the copyright holder. To view a copy of this license visit httpcreativecommons.orglicensesby4.0.",
        r"This is an open access article under the terms of the",
    ]

    regexes = [
        r"License which permits use distribution and reproduction in any medium provided the original work is properly cited.",
        r"Open AccessThis article is distributed under the terms of the Creative Commons Attribution 4.0 International License ",
        r"which permits unrestricted use distribution and reproduction in any medium provided you give appropriate credit to the original authors and the source provide a link to the Creative Commons license and indicate if changes were made. The Creative Commons Public Domain Dedication waiver httpcreativecommons.orgpublicdomainzero1.0 applies to the data made available in this article unless otherwise stated.",
        r"Open Access This article is licensed under a Creative Commons Attribution 4.0 International License which permits use sharing adaptation distribution and reproduction in any medium or format as long as you give appropriate credit to the original authors and the source provide a link to the Creative Commons licence and indicate if changes were made. The images or other third party material in this article are included in the articles Creative Commons licence unless indicated otherwise in a credit line to the material. If material is not included in the articles Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use you will need to obtain permission directly from the copyright holder. To view a copy of this licence visit httpcreativecommons.orglicensesby4.0.",
        r"This is an openaccess article distributed under the terms of the Creative Commons Attribution Noncommercial License which permits unrestricted use distribution and reproduction in any medium provided the original work is properly cited.",
        r"This is an open access article distributed under the terms of the Creative Commons AttributionNon Commercial License 4.0 CCBYNC where it is permissible to download share remix transform and buildup the work provided it is properly cited. The work cannot be used commercially without permission from the journal.",
        r"Open AccessThis article is licensed under a Creative Commons Attribution 4.0 International License which permits use sharing adaptation distribution and reproduction in any medium or format as long as you give appropriate credit to the original authors and the source provide a link to the Creative Commons licence and indicate if changes were made. The images or other third party material in this article are included in the articles Creative Commons licence unless indicated otherwise in a credit line to the material. If material is not included in the articles Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use you will need to obtain permission directly from the copyright holder. To view a copy of this licence visit httpcreativecommons.orglicensesby4.0.",
        r"This article is distributed under the terms of the Creative Commons Attribution 4.0 License httpscreativecommons.orglicensesby4.0 which permits any use reproduction and distribution of the work without further permission provided the original work is attributed as specified on the SAGE and Open Access pages httpsus.sagepub.comenusnamopenaccessatsage.",
        r" httpsorcid.org0000000314504683",
        r"This is an Open Access article distributed under the terms of the Creative Commons Attribution License httpcreativecommons.orglicensesby4.0 which permits unrestricted use distribution and reproduction in any medium provided the original work is properly cited.",
        r"CC BYNC 4.0 License which permits unrestricted noncommercial use distribution and reproduction in any medium provided the original work is properly cited.",
        r"This is an Open Access article distributed under the terms of the Creative Commons Attribution License  which permits unrestricted use distribution and reproduction in any medium provided the original work is properly cited.",
        r"This work is licensed under the Creative Commons AttributionNonCommercialNo Derivative Works 3.0 Unported License. To view a copy of this license visit",
        r"This is an Open Access article distributed under the terms of the Creative Commons Attribution NonCommercial License httpcreativecommons.orglicensesbync4.0 which permits noncommercial reuse distribution and reproduction in any medium provided the original work is properly cited. For commercial reuse please contact journals.permissionsoup.com",
        r"This is an Open Access article distributed under the terms of the Creative Commons Attribution License httpcreativecommons.orglicensesby4.0 which permits unrestricted use distribution and reproduction in any medium provided the original work is properly cited.",
        r"This is an Open Access article distributed under the terms of the Creative Commons Attribution NonCommercial License httpscreativecommons.orglicensesbync4.0 which permits noncommercial reuse distribution and reproduction in any medium provided the original work is properly cited. For commercial reuse please contact journals.permissionsoup.com",
        r"This is an Open Access article distributed under the terms of the Creative Commons Attribution License  which permits unrestricted use distribution and reproduction in any medium provided the original work is properly cited.",
        r"This is an openaccess article distributed under the terms of the Creative Commons Attribution \d\.0 International license which permits unrestricted use distribution and reproduction in any medium provided that the original work is properly attributed.",
        r"This work is published by Dove Medical Press Limited and licensed under Creative Commons Attribution",
        r"Non Commercial unported v3.0 License",
        r"The full terms of the License are available at",
        r"Noncommercial uses of the work are permitted without any further permission from Dove Medical Press Limited provided the work is properly attributed.",
        r"https?:\/\/?creativecommons\.org\/?licenses\/?by(?:-nc|-nd)?(?:-nc)?\/?\d\.\d",
        r"https?://creativecommons\.org/(?:licenses|publicdomain)/by(?:-nc|-nd)?(?:-zero)?/\d\.\d",
        r"This work is licensed under a Creative Commons AttributionNonCommercialNoDerivs 3.0 Unported License",
        r"Copyright \d\d\d\d the Authors. Published by Wolters Kluwer Health Inc.",
        r"httpscreativecommons.orglicensesbync4.0",
        r"This is an openaccess article .*?\. ",
        r"This manuscript version is made available under the CCBYNCND \d.0 license",
        r"This article is distributed under the terms .*?\. ",
        r"This is an Open Access article .*?\. ",
        r"^httpcreative.*?\.0$.",
        r"^httpscreative.*?\.0$.",
        r"This is an openaccess article distributed under the terms of the Creative Commons AttributionNon Commercial License 4.0 CCBYNC where it is permissible to download share remix transform and buildup the work provided it is properly cited. The work cannot be used commercially without permission from the journal.",
        r"0004",
        r"0001",
        r"Usage and distribution for commercial purposes as well as any distribution of modified material requires written permission.",
        r"This article is distributed under the terms of the Creative Commons.*?\.",
        r"This is an open access article .*?\.",
        r"This is an openaccess article distributed under the terms of the Creative Commons AttributionNon Commercial License .*?\.",
        r"The work cannot be used commercially without permission from the journal.",
        r"This article is licensed under the Creative Commons .*?\.",
        r"httpscreativecommons.orglicensesbyncnd4.0",
        r"httpsorcid\.org\d+[A-Z]?\d*",
        r"This is an open access article distributed in accordance with the Creative Commons Attribution Non Commercial CC BYNC 4.0 license which permits others to distribute remix adapt build upon this work noncommercially and license their derivative works on different terms provided the original work is properly cited appropriate credit is given any changes made indicated and the use is noncommercial.",
        r"License which permits use and distribution in any medium provided the original work is properly cited the use is noncommercial and no modifications or adaptations are made.",
        r"This article is distributed under the terms of the Creative CommonsAttribution 4.0 License  which permitsany use reproduction and distribution of the work without furtherpermission provided the original work is attributed as specified on the SAGEand Open Access pages httpsus.sagepub.comenusnamopenaccessatsage.",
        r"Usage and distribution for commercial purposes as well as any distribution of modified material requires written permission.",
        r"The Creative Commons Public Domain Dedication waiver httpcreativecommons.orgpublicdomainzero1.0 applies to the data made available in this article unless otherwise stated in a credit line to the data.",
    ]

    polishes = [
        r"Open AccessThis article is licensed under a Creative Commons AttributionNonCommercial 4.0 International License which permits any noncommercial use sharing adaptation distribution and reproduction in any medium or format as long as you give appropriate credit to the original authors and the source provide a link to the Creative Commons licence and indicate if changes were made. The images or other third party material in this article are included in the articles Creative Commons licence unless indicated otherwise in a credit line to the material. If material is not included in the articles Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use you will need to obtain permission directly from the copyright holder. To view a copy of this licence visit httpcreativecommons.orglicensesbync4.0.",
        r"Creative Commons AttributionNonCommercial",
        r"This article is an openaccess article which was selected by an inhouse editor and fully peerreviewed by external reviewers. It is distributed in accordance with the Creative Commons Attribution Non Commercial CC BYNC 4.0 license which permits others to distribute remix adapt build upon this work noncommercially and license their derivative works on different terms provided the original work is properly cited and the use is noncommercial.",
        r"httpcreativecommons.orglicensesbyncnd3.0",
        r"httpcreativecommons.orglicensesbyncnd4.0",
        r"Neurosciences is an Open Access journal and articles published are distributed under the terms of the Creative Commons AttributionNonCommercial License CC BYNC. Readers may copy distribute and display the work for noncommercial purposes with the proper citation of the original work.",
        r"This article is licensed under a Creative Commons Attribution 4.0 International License which permits use sharing adaptation distribution and reproduction in any medium or format as long as you give appropriate credit to the original authors and the source provide a link to the Creative Commons license and indicate if changes were made.The images or other third party material in this article are included in the articles Creative Commons license unless indicated otherwise in a credit line to the material.If material is not included in the articles Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use you will need to obtain permission directly from the copyright holder. To view a copy of this license visit .",
        r"Creative Commons Non Commercial CC BYNC",
        r"reuse permitted under CC BYNC.",
        r"No commercial reuse. See rights and permissions.",
        r"Published by BMJ.",
        r"Fax\s\d{6,14}",
        r"license which permits others to distribute remix adapt build upon this work noncommercially and license their derivative works on different terms provided the original work is properly cited appropriate credit is given any changes made indicated and the use is noncommercial.",
        r"^\s*(Objective|Background|Abstract|SUMMARY|Purpose)",
        r"usepackageamsfonts",
        r"usepackageamssymb",
        r"supplementary",
    ]
    for pattern in patterns:
        df["text"] = df["text"].str.replace(pattern, "", regex=True)
    for regex in regexes:
        df["text"] = df["text"].str.replace(regex, "", regex=True)
    for polish in polishes:
        df["text"] = df["text"].str.replace(polish, "", regex=True)
    df["text"] = df["text"].str.replace("httpscreativecommons.orglicensesby4.0", "")
    df["text"] = df["text"].str.replace("httpcreativecommons.orglicensesby4.0", "")
    df["text"] = df["text"].str.replace("httpcreativecommons.orglicensesbync3.0", "")
    df["text"] = df["text"].str.replace("^\s*BACKGROUND", "")
    df["text"] = df["text"].str.replace("^\s*Background", "")
    df["text"] = df["text"].str.replace("^\s*Highlights", "")
    df["text"] = df["text"].str.replace("^\s*Objective", "")
    df["text"] = df["text"].str.replace("^\s*Abstract", "")
    df["text"] = df["text"].str.replace("^\s*Purpose", "")
    df["text"] = df["text"].str.replace("^\s*Objectives", "")
    df["text"] = df["text"].str.replace("^\s*OBJECTIVE", "")
    df["text"] = df["text"].str.replace("^\s*Summary", "")
    df["text"] = df["text"].str.replace("^\s*Key Points", "")
    df["text"] = df["text"].str.replace("^\s*ObjectiveBackground", "")
    df["text"] = df["text"].str.replace(
        r"https?:\/\/?creativecommons\.org\/?licenses\/?by(?:-nc|-nd)?\/?\d\.\d", ""
    )
    df["text"] = df["text"].str.replace(
        "This is an Open Access article distributed under the terms of the Creative Commons Attribution NonCommercial License httpcreativecommons.orglicensesbync4.0 which permits noncommercial reuse distribution and reproduction in any medium provided the original work is properly cited. For commercial reuse please contact journals.permissionsoup.com",
        "",
    )
    df["text"] = df["text"].str.replace(r"\sfig\sure", "\sfigure")

    df.drop(df[df["text"].str.len() < 50].index, inplace=True)
    df.to_csv(base_url + "scientific_pmc_cleaned.csv", index=False)


def clean_webmd(base_url, input):

    df = pd.read_csv(base_url + input)
    print("loaded WedMD data")
    df["text"] = df["text"].apply(lambda x: x.split("Related Topics")[0])
    df["text"] = df["text"].apply(lambda x: x.split("Search Related Links")[0])

    def extract_text_1(x):
        try:
            return re.split(
                r"MD on (January|February|March|April|May|June|July|August|September|October|November|December) \d{2} \d{4}",
                x,
            )
            if len(parts) > 1:
                return parts[1]  # Return the desired part
            else:
                return None  # Return None if the pattern does not match
        except Exception as e:
            print(f"Error processing text: {x}, Error: {e}")
            return None  # Return None in case of error

    df["text"] = df["text"].apply(extract_text_1)

    def extract_text_4(x):
        try:
            return re.replace(
                to_replace=r"MD"
                r" on (January|February|March|April|May|June|July|August|September|October|November|December) \d{2} \d{4}",
                value="",
                regex=True,
            )
        except Exception:
            return x

    def extract_text_5(x):
        x = x[0]
        try:
            cleaned_text = re.sub(
                r"\s*View All ADD.*?Written by", "", x, flags=re.DOTALL | re.IGNORECASE
            )
            return cleaned_text
        except Exception as e:
            print(f"Error processing text: {e}")
            return x

    df["text"] = df["text"].apply(extract_text_4)
    df["text"] = df["text"].apply(extract_text_5)

    def extract_text_7(x):
        try:
            cleaned_text = re.sub(
                r"View All ADDADHDAllergiesArthritisAtrial fibrillationBreast CancerCancerCrohns DiseaseDepressionDiabetesDVTEczemaEye HealthHeart DiseaseHIV  AIDSLung DiseaseLupusMental HealthMultiple",
                "",
                x,
            )
            return cleaned_text
        except Exception as e:
            print(f"Error processing text: {e}")
            return x

    def extract_text_8(x):
        try:
            cleaned_text = re.sub(
                r"SclerosisMigrainePain ManagementPsoriasisPsoriatic ArthritisRheumatoid ArthritisSexual ConditionsSkin ProblemsSleep DisordersUlcerative Colitis View All Drugs  Supplements  Back Drugs  SupplementsDrugsSupplementsPill IdentifierInteraction CheckerWellBeing  Back WellBeing View All Aging WellBabyBirth ControlChildrens",
                "",
                x,
            )
            print(f"After  cleaning: {cleaned_text[:100]}...")
            return cleaned_text
        except Exception as e:
            print(f"Error processing text: {e}")
            return x

    def extract_text_9(x):
        try:
            print(f"Before cleaning: {x[:100]}...")
            cleaned_text = re.sub(
                r"HealthDiet  Weight ManagementFitness  ExerciseFood  RecipesHealthy BeautyMens HealthParentingPet HealthPregnancySex  RelationshipsTeen HealthWomens Health View All Symptom CheckerFind a DoctorMore",
                "",
                x,
            )
            return cleaned_text
        except Exception as e:
            print(f"Error processing text: {e}")
            return x

    def extract_text_10(x):
        try:
            cleaned_text = re.sub(
                r"Back MoreNewsBlogsPodcastsWebinarsNewslettersWebMD MagazineBest HospitalsSupport GroupsOrthopedics Privacy  More  Subscribe  Log In  Search  Subscribe",
                "",
                x,
            )
            return cleaned_text
        except Exception as e:
            print(f"Error processing text: {e}")
            return x

    df["text"] = df["text"].apply(extract_text_7)
    df["text"] = df["text"].apply(extract_text_8)
    df["text"] = df["text"].apply(extract_text_9)
    df["text"] = df["text"].apply(extract_text_10)

    def extract_text_11(x):
        try:
            cleaned_text = re.sub(
                r"MenopauseMenopause GuidePerimenopauseMenopause OverviewCauses and SymptomsDiagnosis and TestsProtecting Your HealthPostmenopauseTreatmentLiving With Daily LifeDiet and ExerciseSex and SleepSupport  Resources View Full Guide",
                "",
                x,
            )
            return cleaned_text
        except Exception as e:
            print(f"Error processing text: {e}")
            return x

    df["text"] = df["text"].apply(extract_text_11)

    def extract_text_12(x):
        try:
            cleaned_text = re.sub(
                r"MenopauseReferenceMenopause GuidePerimenopauseMenopause OverviewCauses and SymptomsDiagnosis and TestsProtecting Your HealthPostmenopauseTreatmentLiving With Daily LifeDiet and ExerciseSex and SleepSupport  Resources View Full Guide",
                "",
                x,
            )
            return cleaned_text
        except Exception as e:
            print(f"Error processing text: {e}")
            return x

    df["text"] = df["text"].apply(extract_text_12)

    def extract_text_13(x):
        try:
            cleaned_text = re.sub(
                r"Dementia and AlzheimersAlzheimers Disease  Other Dementias GuideOverview DementiaAlzheimers DiseaseOther DementiasDiagnosis  StagesCauses  Risks OverviewRace As a Risk FactorTreatment Drug TreatmentComplementary  Alternative TreatmentResearch and Potential TreatmentsSymptom ManagementLiving With  Daily LifeDiet  ExerciseSafety IssuesComplications  Related Conditions When Alzheimers Affects More than MemoryPhysical ProblemsSleep ProblemsBehavioral  Cognitive ProblemsGastrointestinal ProblemsPsychosisCaregiving Caregiving EssentialsCaregiver Stress  SelfCareLegal  Financial MattersSupport  Resources Groups  OrganizationsCare Facilities View Full Guide",
                "",
                x,
            )
            return cleaned_text
        except Exception as e:
            print(f"Error processing text: {e}")
            return x

    df["text"] = df["text"].apply(extract_text_13)
    df["text"] = df["text"].str.replace(r"StrokeWhat", "Stroke What", regex=True)

    patterns = [
        "^.*?View Full Guide",
        "",
        "^.*?From the WebMD Archives",
        "",
        "Reviewed by .*?(?=[.!?]|$)",
        "",
        "View All ADD ADHDAllergiesArthritisAtrial fibrillationBreast CancerCancerCrohn s DiseaseDepressionDiabetesDVTEczemaEye HealthHeart DiseaseHIV   AIDSLung DiseaseLupusMental HealthMultiple SclerosisMigrainePain ManagementPsoriasisPsoriatic ArthritisRheumatoid ArthritisSexual ConditionsSkin ProblemsSleep DisordersUlcerative Colitis ",
        "(?<=View All).*?(?=View All)",
        "(?<=View All).*?(?=View Full Guide)",
        "(?<=View All).*?(?=min read)",
        "policyeditorial.*",
        "videosfind.*",
        "This site is protected by recaptcha.*",
        "Sources  Update.*",
        "TopicsPoliciesPrivacy PolicyCookie",
        "PolicyCookie",
        "httpswww.webmd.com",
        "WebMD LLC.",
        "Find more articles browse back issues and read the current issue of WebMD Magazin",
        "See additional information.",
        "WebMD does not provide medical advice diagnosis or treatment.",
        "20052024 WebMD LLC. All rights reserved.",
        "Advertise With Us  Terms of Use  Privacy Policy  Cookie Policy  Editorial Policy  Contact Us  AdChoice",
        "About WebMD",
        "Go Now",
        "Load More See More on (?:\{[^}]+\}|[^ ]+ (?:[^ ]+ )?[^ ]+) From WebMD",
        "Skip to main content",
        "Home Conditions  Back Conditions",
        "View All Drugs   Supplements  Back Drugs",
        "SupplementsDrugsSupplementsPill IdentifierInteraction CheckerWell Being  Back Well Being View All",
        "Aging WellBabyBirth ControlChildren s HealthDiet   Weight ManagementFitness   ExerciseFood   RecipesHealthy BeautyMen s HealthParentingPet HealthPregnancySex",
        "RelationshipsTeen HealthWomen s Health View All Symptom CheckerFind a DoctorMore",
        "Back MoreNewsBlogsPodcastsWebinarsNewslettersWebMD MagazineBest HospitalsSupport GroupsOrthopedics",
        "Privacy   More  Subscribe  Log In  Search",
        "Hide Video Transcript",
        "Video Transcript",
        "MUSIC PLAYING",
        "Health ServicesSite",
        "See additional information.",
        "WebMD does not provide medical advice  diagnosis or treatment.",
        "AppPregnancyBabyAllergyFor",
        "AdvertisersAdvertise with UsAdvertising Policy",
        "MapAccessibilityOur",
        "WebMD LLC  an Internet Brands company.",
        "AppsWebMD",
        "MobileWebMD",
        "Show Sources Share View privacy policy and trust info",
        "All rights reserved.",
        "Recommended FEATURED Top doctors in",
        "Find more top doctors on",
        "Search PoliciesPrivacy",
        "PolicyCookie",
        "PolicyEditorial",
        "PolicyAdvertising",
        "PolicyCorrection",
        "PolicyTerms of",
        "UseAboutContact",
        "UsAbout",
        "WebMDCareersNewsletterCorporateWebMD",
        "Updates in Your InboxWebMD CareFind a Telemedicine Doc",
        "WebMD Care",
        "WebMD Editorial Contributors",
    ]

    for pattern in patterns:
        df["text"] = df["text"].str.replace(pattern, "", regex=True)

    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() >= 150]
    df.to_csv(base_url + "vernacular_WebMed_cleaned.csv", index=False)


def clean_harvardHealthPublishing(base_url, input):

    df = pd.read_csv(base_url + input)
    df = df.drop_duplicates()
    print("loaded Harvard Health Publishing data")
    df["text"] = df["text"].str.replace("Harvard Health Publishing", "")
    df["text"] = df["text"].str.replace("Harvard Health", "")
    df["text"] = df["text"].apply(
        lambda x: x.split("This site is protected by reCAPTCHA")[0]
    )
    df["text"] = df["text"].apply(lambda x: re.split("About the Author.*", x)[0])

    patterns = [
        "(?<=Search).*?(?=Common Conditions)",
        "Search",
        "Common Conditions",
        "(?<=Executive Editor).*?(?=Editor in Chief,)",
        "Executive Editor",
        "Editor in Chief",
        "(?<=, Harvard).*?(?=Editor,)",
        "Editorial Advisory Board Member",
        "Reviewed by",
        "Health Writer ",
        ", Harvard Women's Health WatchReviewed by",
        ", Harvard Men's Health Watch",
        "Open mobile menu",
        "Recent Blog Articles",
        "Staying Healthy      COVID-19 Updates       Close menu    Close    Main Content",
        "Dupuytren's contracture of the hand        Moving from couch to 5K        How — and why — to fit more fiber and fermented food into your meals        Tick season is expanding: Protect yourself against Lyme disease        What? Another medical form to fill out?        How do trees and green spaces enhance our health?",
        "  A muscle-building obsession in boys: What to know and do         Ad Watch: New drug, old song, clever tagline        Concussion in children: What to know and do        What color is your tongue? What's healthy, what's not?      / ",
        "Contributor\\; ",
        "Harvard Women's Health Watch",
        "Harvard Heart Letter",
        "Sign Me Up",
        "Free Healthbeat Signup",
        "Already a member? Login ».",
        "Staying Healthy",
        "Helpful Links",
        "Shop",
        "Pay Subscription Bill" "Close          Shopping Cart",
        "About Us",
        "Online Courses",
        "Blog",
        "Resources",
        "Special Health Reports",
        "Subscriptions",
        "Login",
        "Content Licensing" "Customer Service",
        "(?<=Close).*?(?=Main Content)",
        "(?<=CloseShopping Cart).*?(?=What's healthy, what's not?)",
        "(?<=Recent Blog Articles).*?(?=Terms of Service apply.)",
        "Log In",
        "Open mobile menu",
        "Pay Subscription Bill",
        "COVID-19 Updates",
        "Close menu",
        "Main Content",
        "Menu",
        "Find more top doctors on",
        "Top doctors in",
        "Alzheimers Disease BasicsAlzheimers vs. Dementia Understanding the Differences",
        "DiseaseUnderstanding",
        "Brain Foods That May Help Prevent Dementia",
        "Recommended FEATURED",
    ]
    df["text"] = df["text"].str.strip()

    for pattern in patterns:
        df["text"] = df["text"].str.replace(pattern, "", regex=True)

    df["text"] = df["text"].astype(str)
    df.drop(df[df["text"].str.len() < 50].index, inplace=True)
    df.to_csv(base_url + "vernacular_HHP_cleaned.csv", index=False)


def clean_MH(base_url, input):

    df = pd.read_csv(base_url + input)
    df = df.drop_duplicates()
    print("loaded Mens health data")

    def extract_text_2(x):
        try:
            return re.split(r"sign inGuide to Therapy", x)[1]
        except Exception:
            return x

    df["text"] = df["text"].apply(extract_text_2)

    def extract_text_8(x):
        try:
            return re.split(
                r"Best Running ShortsBiohack Your SkinBest Food ScalesMuscle Building at 50",
                x,
            )[1]
        except Exception:
            return x

    df["text"] = df["text"].apply(extract_text_8)

    patterns = [
        "Men's Health",
        "Subscribe to Men's Health",
        "Shop at Amazon",
        "Watch Next",
        "Related Story",
        "Published: [A-Za-z]{3} \d{1,2}, \d{4} \d{1,2}:\d{2} [AP]M ESTSave Article",
        "Our product picks are editor-tested, expert-approved. We may earn a commission through links on our site. Why Trust Us?",
        "SearchAbout",
        "My BookmarksMVP",
        "ExclusivesShopHealthFitnessWorkoutsWeight Loss",
        "Entertainment",
        "Sex & Relationships",
        "LifeTechnology & GearStyle",
        "NutritionGrooming Awards",
        "VideoNewsletterFollowPromotionsSubscribeOther",
        "EditionsPrivacy",
        "NoticeTerms Of UseSkip to Content",
        "FitnessHealthGearGroomingShopping",
        "Subscribesign in",
        "Best Walking Workouts",
        "Marvel Movie Ranking",
        "Best Chinos",
        "SearchBest Beard Trimmers",
        " Deadlift vs. Squat",
    ]

    for pattern in patterns:
        df["text"] = df["text"].str.replace(pattern, "", regex=True)
    df["text"] = df["text"].astype(str)
    df.drop(df[df["text"].str.len() < 50].index, inplace=True)
    df.to_csv(base_url + "vernacular_MH_cleaned.csv", index=False)


def clean_WH(base_url, input):

    df = pd.read_csv(base_url + input)
    df = df.drop_duplicates()
    print("loaded Women Health data")
    patterns = [
        "Women's Health",
        "Watch Next",
        "SubscribeMy",
        "EditionsPrivacy",
        "NoticeTerms Of Use",
        "Skip to ContentHealthFitnessBeautyLifeRelationshipsSubscribesign",
        "Best Fitness Trackers",
        "30-Min Full-Body Workout",
        "Walking Shoes for Women" "Workout Finder" "SearchAbout",
        "LoveRelationshipsLifeAwardsNewsletterFollowShopOther",
        "Bookmarks+FitnessHealthBeautyFoodSports & AthletesStyleWeight LossSex &amp;",
    ]
    for pattern in patterns:
        df["text"] = df["text"].str.replace(pattern, "", regex=True)

    df["text"] = df["text"].astype(str)
    df.drop(df[df["text"].str.len() < 50].index, inplace=True)
    df.to_csv(base_url + "vernacular_WH_cleaned.csv", index=False)


def clean_MedlinePlus(base_url, col_list, input):
    df = pd.read_csv(base_url + input)
    df = df.drop_duplicates()
    print(print("loaded MedlinePlus data"))
    df["text"] = df["text"].replace(
        to_replace="(?<=NIH MedlinePlus Magazine).*?(?=Espaol)", value="", regex=True
    )
    df["text"] = df["text"].str.replace(
        "NIH MedlinePlus MagazineEspaol     Espaol  rss facebook  youtube        Search  Search                  Health AZ    Anxiety   Antidepressants   Breast Cancer   Cholesterol   COVID19   Hypothyroidism   Palliative Care   Physical Activity   Skin Conditions   View all topics      NIH Research    Research Highlights   NIH Technology Breakthroughs   Meet the Researchers   Resources at NIH      Issues    Current Issue   Past Issues   Archived Issues      Multimedia    Video   Infographics   Health Fast Facts   All Multimedia      About    Contact us      Subscribe    Email Updates         Search  Search          ",
        "",
    )
    df["text"] = df["text"].astype(str)

    def extract_text_3(x):
        try:
            return re.split(r"Sources MedlinePlus", x)[0]
        except Exception:
            return x

    df["text"] = df["text"].apply(extract_text_3)

    def extract_text_6(x):
        try:
            return re.split(r"MedlinePlus Delivered", x)[0]
        except Exception:
            return x

    df["text"] = df["text"].apply(extract_text_6)

    def extract_text_7(x):
        try:
            return re.split(r"Explore More on MedlinePlus.gov", x)[0]
        except Exception:
            return x

    df["text"] = df["text"].apply(extract_text_7)

    def extract_text_8(x):
        try:
            return re.split(
                r"MedlinePlus offers reliable uptodate health information anytime anywhere for free.",
                x,
            )[0]
        except Exception:
            return x

    df["text"] = df["text"].apply(extract_text_8)

    df.drop(df[df["text"].str.len() < 50].index, inplace=True)
    df.to_csv(base_url + "vernacular_MedlinePlus_cleaned.csv", index=False)


def clean_JEBIM(base_url, col_list, input):
    df = pd.read_csv(base_url + input)
    df = df.drop_duplicates()
    patterns = [
        (r"Original Article", ""),
        (r"Topical Review Article", ""),
        (r"Review Article", ""),
        (
            r"Creative Commons Non Commercial CC BYNC This article is distributed under the terms of the Creative Commons AttributionNonCommercial 4\.0 License https://creativecommons\.org/licenses/by-nc/4\.0 which permits noncommercial use, reproduction and distribution of the work without further permission provided the original work is attributed as specified on the SAGE and Open Access pages https://us\.sagepub\.com/en-us/nam/open-access-at-sage",
            "",
        ),
        (
            r"Creative Commons Non Commercial CC BYNC This article is distributed under the terms of the Creative Commons AttributionNonCommercial 4\.0 License http://www\.creativecommons\.org/licenses/by-nc/4\.0 which permits noncommercial use, reproduction and distribution of the work without further permission provided the original work is attributed as specified on the SAGE and Open Access pages https://us\.sagepub\.com/en-us/nam/open-access-at-sage",
            "",
        ),
        (
            r"Creative Commons Non Commercial CC BYNC This article is distributed under the terms of the Creative Commons AttributionNonCommercial 3\.0 License http://www\.creativecommons\.org/licenses/by-nc/3\.0 which permits noncommercial use, reproduction and distribution of the work without further permission provided the original work is attributed as specified on the SAGE and Open Access pages https://us\.sagepub\.com/en-us/nam/open-access-at-sage",
            "",
        ),
        (
            r"Creative Commons Non Commercial CC BY NC This article is distributed under the terms of the Creative Commons AttributionNonCommercial 4\.0 License https://creativecommons\.org/licenses/by-nc/4\.0 which permits noncommercial use, reproduction and distribution of the work without further permission provided the original work is attributed as specified on the SAGE and Open Access page https://us\.sagepub\.com/en-us/nam/open-access-at-sage\. Original Manuscript Journal of Evidence-Based Integrative Medicine",
            "",
        ),
        (r"journals\.sagepub\.com/home/cam", ""),
        (r"Springer Publishing Company", ""),
        (r"Reprints and permission sagepub\.com/journalsPermissions\.nav", ""),
        (r"The Authors", ""),
        (r"cantly ", "cantly "),
        (r"\s+sion\s+", "sion "),
        (r"[ ]{1,2}sion ", "sion "),
        (r"tive ", "tive "),
        (r"http://www\.biomedcentral\.com", ""),
        (r"http ", ""),
        (r"ammatory ", "ammatory "),
    ]

    regexes = [
        (r" signi ", " signi "),
        (r"https://us\.sagepub\.com/en-us/nam/open-access-at-sage", ""),
        (r" nThe", " The"),
        (r" nthe", " the"),
        (r" nand ", " and "),
        (r" nof ", " of "),
        (r" nIn ", " In "),
        (r" nto ", " to "),
        (r"nwith", "with"),
        (r"nwere", "were"),
        (r"nfor", "for"),
        (r"nwas", "was"),
        (r"ntion", "tion"),
        (r"nResults", "Results"),
        (r"nMethods", "Methods"),
        (r"nFig", "Fig"),
        (r"ngroup", "group"),
        (r"nthat", "that"),
        (r"nstudy", "study"),
        (r"ntreatment", "treatment"),
        (r"nhttp", " "),
        (r"nTable", "Table"),
        (r"nBackground", "Background"),
        (r"nThis", "This"),
        (r"nfrom", "from"),
        (r"nmedicine", "medicine"),
        (r"nment", "ment"),
        (r"nby", "by"),
        (r"nWe", "We"),
        (r"nacupuncture", "acupuncture"),
        (r"npatients", "patients"),
        (r"nResults", "Results "),
        (r"nBMC", "BMC"),
        (r"nNo", "No"),
        (r"nConclusions", "Conclusions"),
        (r"nConclusion", "Conclusion"),
        (r"ndata", "data"),
        (r"nafter", "after"),
        (r"nfigure", "figure"),
        (r"neffects", "effects"),
        (r"neffect", "effect"),
        (r"nstudies", "studies"),
        (r"[ ]{1,2}tion", "tion"),
        (r"[ ]{1,2}tions", "tions"),
        (r"[ ]{1,2}ing", "ing"),
        (r"\s+ing\s+", "ing "),
        (r"[ ]{1,2}nture ", "ture "),
        (r"nin", "in"),
        (r"nAll", "All"),
        (r"nreceived", "received"),
        (r"nReceived", "Received"),
        (r"prevetion", "prevention"),
        (r"nwhich", "which"),
        (r"nused", "used"),
        (r"nuse ", "use "),
        (r"nusing", "using"),
        (r"nstroke", "stroke"),
        (r"nStroke", "Stroke"),
        (r"nTo", "To"),
        (r"nAuthors", "Authors"),
        (r"nAuthor", "Author"),
        (r"nauthor", "author"),
        (r"nanalysis ", "analysis "),
        (r"nAnalysis ", "Analysis "),
        (r"nDiscussion ", "Discussion "),
        (
            r"nSubmit your next manuscript to BioMed Central and take full advantage of  n  Convenient online submission n  Thorough peer review n  No space constraints or color figure charges n  Immediate publication on acceptance n  Inclusion in PubMed, CAS, Scopus and Google Scholar n  Research which is freely available for redistribution nSubmit your manuscript at  submit",
            "",
        ),
        (r"learning ", "learning "),
        (r"training ", "training "),
        (r"into ", "into "),
        (r"conventional ", "conventional "),
        (r"intervention ", "intervention "),
        (r"interventions ", "interventions "),
        (r"information", "information"),
        (r"informed", "informed"),
        (r"nReferences", "References"),
        (r"nMedicine", "Medicine"),
        (r"nare ", "are "),
        (r"nnot ", "not "),
        (r"nour ", "our "),
        (r"nhave ", "have "),
        (r"nsubmit ", "submit "),
        (r"nhealth", "health"),
        (r"nAbstract ", "Abstract "),
        (r"nCorrespondence ", "Correspondence "),
        (r"nCompeting ", "Competing "),
        (r"nAdditional ", "Additional "),
        (r"nKeywords ", "Keywords "),
        (r"ncontrol", "control"),
        (r"Iformed", "informed"),
        (r"iformation", "information"),
        (r"intervetions", "interventions"),
        (r"convetional", "conventinal"),
        (r"ncompared ", "compared "),
        (r"nStatistical ", "Statistical "),
        (r"nstatistical ", "statistical "),
        (r"[ ]{1,2}ntive ", "tive "),
        (r"nPage \d{1,3} of \d{1,3}", ""),
        (r"nbe ", "be "),
        (r"ncells ", "cells "),
        (r"nmore ", "more "),
        (r"significant", "significant"),
        (r"nage", "age"),
        (r"nAge", "Age"),
        (r"dam age ", "damage "),
        (r"percent age ", "percentage "),
        (r"nsion", "sion"),
        (r"nli", "li"),
        (r"nbetween ", "between "),
        (r"nexpression ", "expression "),
        (r"nuniversity ", "university "),
        (r"nUniversity ", "University "),
        (r"nmodel ", "model "),
        (r"nanti ", "anti "),
        (r"Annotated Entity\s+ID\s+\d+\s+Spans\s+True\s+Boxes\s+True\s+Text", ""),
        (r"ANnotated Entity\s+ID\s+\d+\s+Spans\s+True\s+Boxes\s+True\s+Text", ""),
        (r"ANotated Entity ID \d+ Spans True Boxes True Text", ""),
        (r"Abstract", ""),
        (r"Background", ""),
        (r"npage", ""),
        (r"nthe", ""),
        (r"ntable", "table"),
        (r"nresults", "results"),
        (r"nbmc", "bmc"),
        (r"nmethods", "methods"),
        (r"nfig ure", "figure"),
        (r"nfigure", "figure"),
        (r"fig ure", "figure"),
        (r"nbackground", ""),
        (r"nconclusions", ""),
        (r"nthis", "this"),
        (r"nwe", "we"),
        (r"nafter", "after"),
        (r"oline", "online"),
        (" ment", "ment"),
        (r" tion", "tion"),
        (r" hypertesion ", " hypertension "),
    ]

    df["text"] = df["text"].astype(str)
    for pattern, replacement in patterns:
        df["text"] = df["text"].str.replace(
            pattern, replacement, regex=True, flags=re.IGNORECASE
        )

    for regex, replacement in regexes:
        df["text"] = df["text"].str.replace(regex, "", regex=True)

    df.drop(df[df["text"].str.len() < 50].index, inplace=True)
    df.to_csv(base_url + "alternative_JEBIM_cleaned.csv", index=False)


def clean_CMT(base_url, input):
    df = pd.read_csv(base_url + input)
    df = df.drop_duplicates()
    df["text"] = df["text"].str.split(
        r"BMC Complementary and Alternative Medicine \d{4}", expand=True
    )[0]

    patterns = [
        (r"Annotated Entity\s+ID\s+\d+\s+Spans\s+True\s+Boxes\s+True\s+Text", ""),
        (r"RESEARCH ARTICLE Open Access", ""),
        (r"RES EAR CH A RT I C LE nOpen Access n", ""),
        (r"The Authors", ""),
        (
            r"This is an Open Access article distributed under the terms of the Creative Commons Attribution License https://creativecommons.org/licenses/by/2\.0 which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited.",
            "",
        ),
        (r"https://doi.org", ""),
        (r"http ", ""),
        (r"[ ]{1,2}ing ", "ing "),
        (r"\s+ing\s+", "ing "),
        (r"[ ]{1,2}ment ", "ment "),
        (r"[ ]{1,2}nture ", "ture "),
        (r"cantly ", "cantly "),
        (r"\s+sion\s+", "sion "),
        (r"[ ]{1,2}sion ", "sion "),
        (r"usepackageamsfonts", ""),
        (
            r"To view a copy of this nlicence, visit licenses by 4\.0\. The Creative Commons Public Domain Dedication waiver http://creativecommons.org/publicdomain/zero/1\.0 applies to the data made available in this article unless otherwise stated in a credit line to the data.",
            "",
        ),
        (
            r"Open Access This article is licensed under a Creative Commons Attribution 4\.0 International License, which permits use, sharing, adaptation, distribution, and reproduction in any medium or format, as long as you give appropriate credit to the original authors and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article\'s Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article\'s Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder.",
            "",
        ),
        (
            r"To view a copy of this nlicence, visit http://creativecommons.org/licenses/by/4\.0\. The Creative Commons Public Domain Dedication waiver http://creativecommons.org/publicdomain/zero/1\.0 applies to the data made available in this article unless otherwise stated in a credit line to the data.",
            "",
        ),
        (r"https://creativecommons.org", ""),
        (
            r"This is an Open Access article distributed under the terms of the Creative Commons Attribution License",
            "",
        ),
        (r"licenses by 2\.0", ""),
        (
            r"which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited.",
            "",
        ),
        (
            r"Open Access This article is licensed under a Creative Commons Attribution 4\.0 International License, which permits use, sharing, adaptation, distribution, and reproduction in any medium or format, as long as you give appropriate credit to the original authors and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article\'s Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article\'s Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit licenses by 4\.0\. The Creative Commons Public Domain Dedication waiver http://creativecommons.org/publicdomain/zero/1\.0 applies to the data made available in this article unless otherwise stated in a credit line to the data.",
            "",
        ),
        (
            r"Open Access This article is distributed under the terms of the Creative Commons Attribution 4\.0 International License licenses by 4\.0 which permits unrestricted use, distribution, and reproduction in any medium, provided you give appropriate credit to the original authors and the source, provide a link to the Creative Commons license, and indicate if changes were made. The Creative Commons Public Domain Dedication waiver applies to the data made available in this article unless otherwise stated.",
            "",
        ),
        (r"nThe", "The"),
        (r"nthe", "the"),
        (r"nand", "and"),
        (r"nof", "of"),
        (r"nIn", "In"),
        (r"nto", "to"),
        (r"nwith", "with"),
        (r"nwere", "were"),
        (r"nfor", "for"),
        (r"nwas", "was"),
        (r"ntion", "tion"),
        (r"nResults", "Results"),
        (r"nMethods", "Methods"),
        (r"nfig ure", "figure"),
        (r"nfigure", "figure"),
        (r"fig ure", "figure"),
        (r"nFig", "Fig"),
        (r"ngroup", "group"),
        (r"nthat", "that"),
        (r"nstudy", "study"),
        (r"ntreatment", "treatment"),
        (r"nhttp", ""),
        (r"nTable", "Table"),
        (r"nBackground", "Background"),
        (r"nThis", "This"),
        (r"nfrom", "from"),
        (r"nmedicine", "medicine"),
        (
            r"nSubmit your next manuscript to BioMed Central and take full advantage of n Convenient online submission n Thorough peer review n No space constraints or color figure charges n Immediate publication on acceptance n Inclusion in PubMed, CAS, Scopus and Google Scholar n Research which is freely available for redistribution nSubmit your manuscript at submit",
            "",
        ),
        (r"learing ", "learning "),
        (r"traiing ", "training "),
        (r"into ", "into "),
        (r"convetional ", "conventional "),
        (r"intervetion ", "intervention "),
        (r"intervetions ", "interventions "),
        (r"iformation", "information"),
        (r"informed", "informed"),
        (r"hypertesion", "hypertension"),
        (r"maagement", "management"),
        (r"prevetion", "prevention"),
        (r"additioally", "additionally"),
        (r"staiing", "staining"),
        (r"ito", "into"),
        (r"nReferences", "References"),
        (r"nMedicine", "Medicine"),
        (r"nare ", "are "),
        (r"nnot ", "not "),
        (r"nour ", "our "),
        (r"nhave ", "have "),
        (r"nsubmit ", "submit "),
        (r"nhealth", "health"),
        (r"nAbstract ", "Abstract "),
        (r"nCorrespondence ", "Correspondence "),
        (r"nCompeting ", "Competing "),
        (r"nAdditional ", "Additional "),
        (r"nKeywords ", "Keywords "),
        (r"ncontrol", "control"),
        (r"ncompared ", "compared "),
        (r"nStatistical ", "Statistical "),
        (r"nstatistical ", "statistical "),
        (r"nPage \d{1,3} of \d{1,3}", ""),
        (r"nbe ", "be "),
        (r"ncells ", "cells "),
        (r"nmore ", "more "),
        (r"significant", "significant"),
        (r"nage", "age"),
        (r"nAge", "Age"),
        (r"dam age ", "damage "),
        (r"percent age ", "percentage "),
        (r"nsion", "sion"),
        (r"nli", "li"),
        (r"nbetween ", "between "),
        (r"nexpression ", "expression "),
        (r"nuniversity ", "university "),
        (r"nUniversity ", "University "),
        (r"nmodel ", "model "),
        (r"nanti ", "anti "),
        (r"contaiIng", "containing"),
        (r"Annotated Entity  ID  0 Spans  True Boxes  True Text  Abstract", ""),
        (r"Annotated Entity\s+ID\s+\d+\s+Spans\s+True\s+Boxes\s+True\s+Text", ""),
        (r"Copyright\s+\d{4}", ""),
        (r"(?<=\s)nat(?=\s)", ""),
        (r" nNA ", " NA "),
        ("10.1186 s12906", ""),
        ("10.1186", ""),
    ]

    patterns_ = [
        (r"nment", "ment"),
        (r"nby", "by"),
        (r"nWe", "We"),
        (r"nacupuncture", "acupuncture"),
        (r"npatients", "patients"),
        (r"nResults", "Results "),
        (r"nBMC", "BMC"),
        (r" nNo ", " No "),
        (r"nConclusions", "Conclusions"),
        (r"nConclusion", "Conclusion"),
        (r"ndata", "data"),
        (r"nafter", "after"),
        (r"neffects", "effects"),
        (r"neffect", "effect"),
        (r"nstudies", "studies"),
        (r" nin ", " in "),
        (r"nAll", "All"),
        (r"nreceived", "received"),
        (r"nReceived", "Received"),
        (r"prevention", "prevention"),
        (r"nwhich", "which"),
        (r"nused", "used"),
        (r"nuse ", "use "),
        (r"nusing", "using"),
        (r"nstroke", "stroke"),
        (r"nStroke", "Stroke"),
        (r" nTo ", " To "),
        (r"nAuthors", "Authors"),
        (r"nAuthor", "Author"),
        (r"nauthor", "author"),
        (r"nanalysis ", "analysis "),
        (r"nAnalysis ", "Analysis "),
        (r"nDiscussion ", "Discussion "),
        (
            r"This is an openaccess article distributed under the terms of the Creative Commons AttributionNoncommercial License CC BYNC which permits unrestricted use, distribution, and reproduction in any medium provided the original work is properly cited.",
            "",
        ),
        (
            r"This is an openaccess article distributed under the terms of the Creative Commons AttributionNoncommercial License CC BYNC which permits unrestricted use, distribution, and reproduction in any medium provided the original work is properly cited.",
            "",
        ),
        (
            r"This is an openaccess article distributed under the terms of the Creative Commons AttributionNonCommercialNoDerivatives License which permits unrestricted reproduction and distribution for noncommercial purposes only and use and reproduction but not distribution of adapted material for noncommercial purposes only provided the original work is properly cited.",
            "",
        ),
        (
            r"This is an openaccess article distributed under the terms of the Creative Commons Attribution Noncommercial License which permits use, distribution, and reproduction in any medium provided the original work is properly cited, the use is non-commercial, and is otherwise in compliance with the license. See https://creativecommons.org/licenses/by-nc/4.0 and https://creativecommons.org/licenses/by-nc/4.0/legalcode.",
            "",
        ),
        (
            r"This is an Open Access article distributed under the terms of the Creative Commons AttributionNonCommercial License https://creativecommons.org/licenses/by-nc/4.0 which permits unrestricted noncommercial use, distribution, and reproduction in any medium provided the original work is properly cited.",
            "",
        ),
        (
            r"This is an open access article distributed in accordance with the Creative Commons Attribution 4.0 Unported CC BY 4.0 license which permits others to copy, redistribute, remix, transform, and build upon this work for any purpose, provided the original work is properly cited, a link to the license is given, and indication of whether changes were made.",
            "",
        ),
        (r"Reuse permitted under CC BY.", ""),
        (
            r"This is an openaccess article distributed under the terms of the Creative Commons AttributionNon Commercial License 4.0 CCBYNC where it is permissible to download, share, remix, transform, and buildup the work provided it is properly cited. The work cannot be used commercially without permission from the journal.",
            "",
        ),
        (
            r"This article is licensed under a Creative Commons AttributionNonCommercial 4.0 International License which permits any noncommercial use, sharing, adaptation, distribution, and reproduction in any medium or format as long as you give appropriate credit to the original authors and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article\'s Creative Commons license unless indicated otherwise in a credit line to the material. If material is not included in the article\'s Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0.",
            "",
        ),
        (
            r"SAGE Publications Inc. unless otherwise noted. Manuscript content on this site is licensed under Creative Common Licenses",
            "",
        ),
        (
            r"This article is distributed under the terms of the Creative Commons AttributionNonCommercial 4.0 License which permits noncommercial use, reproduction, and distribution of the work without further permission provided the original work is attributed as specified on the SAGE and Open Access pages https://us.sagepub.com/en-us/nam/open-access-at-sage.",
            "",
        ),
        (
            r"This is an Open Access article distributed under the terms of the Creative Commons Attribution license which permits unrestricted reuse, distribution, and reproduction in any medium provided the original work is properly cited.",
            "",
        ),
        (r"This work is licensed under the Creative Commons Attribution License.", ""),
        (
            r"This is an Open Access article distributed under the terms of the Creative Commons AttributionNonCommercial License https://creativecommons.org/licenses/by-nc/4.0 which permits unrestricted noncommercial use, distribution, and reproduction in any medium provided the original work is properly cited.",
            "",
        ),
        (r"mentioned", "mentioned"),
        (r"Abstract", ""),
        (r"Background", ""),
        (r"Introduction", ""),
        (r"npage", ""),
        (r" nthe ", " "),
        (r" ntable ", " table "),
        (r"nresults", "results"),
        (r"nbmc", "bmc"),
        (r"nmethods", "methods"),
        (r"nbackground", ""),
        (r"nconclusions", ""),
        (r"nthis", "this"),
        ("canot", "cannot"),
        (r"nwe", "we"),
        (r"nafter", "after"),
        (r"fig\sure", "figure"),
        (" ment", "ment"),
        (r" iformed", "informed"),
        (r"oline", "online"),
        (r" tion", "tion"),
        (r" morIng ", " morning "),
        (r" burIng ", " burning "),
        (r"medi  ncine", "medicine"),
    ]

    df["text"] = df["text"].astype(str)
    for pattern, replacement in patterns:
        df["text"] = df["text"].str.replace(
            pattern, replacement, regex=True, flags=re.IGNORECASE
        )
    for pat, rep in patterns_:
        df["text"] = df["text"].str.replace(pat, rep, regex=True, flags=re.IGNORECASE)

    df["text"] = df["text"].str.replace("www.biomedcentral.com", "")
    df["text"] = df["text"].str.replace(r"\stion\s", "tion\s")
    df["text"] = df["text"].str.replace(r"\stions\s", "tions\s")
    df["text"] = df["text"].str.replace(r"\sing\s", "ing\s")
    df["text"] = df["text"].str.replace(r"\sment\s", "ment\s")
    df["text"] = df["text"].str.replace(r"maagement", "management")
    df["text"] = df["text"].str.replace(r"hypertesion", "hypertension")
    print(df.head())
    df.drop(df[df["text"].str.len() < 50].index, inplace=True)

    df.to_csv(base_url + "alternative_CMT_cleaned.csv", index=False)


def clean_homeoJ(base_url, input):
    df = pd.read_csv(base_url + input)
    df = df.drop_duplicates()
    print(df.head())
    patterns = [
        (r" httpwww.homoeopathicjournal.com", ""),
        (r"Abstract", ""),
        (r"Annotated Entity\s+ID\s+\d+\s+Spans\s+True\s+Boxes\s+True\s+Text", ""),
        (r"[ ]{1,2}ment ", "ment "),
        (r"[ ]{1,2}ing ", "ing "),
        (r"\s+ing\s+", "ing "),
        (
            r"International Journal of Homeopathic Sciences \d{4} \d{2} \d{4,6} EISSN \d{8} PISSN \d{8} www\.homeopathicjournal\.com",
            "",
        ),
        (r"Received \d{8}", ""),
        (r"Accepted \d{8}", ""),
        (r"cantly ", "cantly "),
        (r"\s+sion\s+", "sion "),
        (r"[ ]{1,2}sion ", "sion "),
        (r"[ ]{1,2}nture ", "ture "),
        (r"http://www\.homeopathicjournal\.com", ""),
        (r"International Journal of Homeopathic Sciences", ""),
        (r"nThe", "The"),
        (r"nthe", "the"),
        (r"nand", "and"),
        (r"nof", "of"),
        (r"nto", "to"),
        (r"nwith", "with"),
        (r"nwere", "were"),
        (r"nfor", "for"),
        (r"nwas", "was"),
        (r"ntion", "tion"),
        (r"nResults", "Results"),
        (r"nBMC", "BMC"),
        (r"nMethods", "Methods"),
        (r"nFig", "Fig"),
        (r"nfig ure", "figure"),
        (r"nfigure", "figure"),
        (r"fig ure", "figure"),
        (r"ngroup", "group"),
        (r"nthat", "that"),
        (r"nstudy", "study"),
        (r"ntreatment", "treatment"),
        (r"nhttp", ""),
        (r"nIn", "In"),
        (r"nResults", "Results "),
        (r"nBackground", "Background"),
        (r"nThis", "This"),
        (r"nfrom", "from"),
        (r"nmedicine", "medicine"),
        (r"nment", "ment"),
        (r"nby", "by"),
        (r"nWe", "We"),
        (r"nacupuncture", "acupuncture"),
        (r"npatients", "patients"),
        (r"nTable", "Table"),
        (r"nNo", "No"),
        (r"nConclusions", "Conclusions"),
        (r"nConclusion", "Conclusion"),
        (r"ndata", "data"),
        (r"nafter", "after"),
        (r"neffects", "effects"),
        (r"neffect", "effect"),
        (r"nstudies", "studies"),
        (r"nin", "in"),
        (r"nAll", "All"),
        (r"nreceived", "received"),
        (r"nReceived", "Received"),
        (r"prevention", "prevention"),
        (r"nwhich", "which"),
        (r"nused", "used"),
        (r"nuse ", "use "),
        (r"nusing", "using"),
        (r"nstroke", "stroke"),
        (r"nStroke", "Stroke"),
        (r"nTo", "To"),
        (r"nAuthors", "Authors"),
        (r"nAuthor", "Author"),
        (r"nauthor", "author"),
        (r"nanalysis ", "analysis "),
        (r"nAnalysis ", "Analysis "),
        (r"nDiscussion ", "Discussion "),
        (
            r"nSubmit your next manuscript to BioMed Central and take full advantage of n Convenient online submission n Thorough peer review n No space constraints or color figure charges n Immediate publication on acceptance n Inclusion in PubMed, CAS, Scopus and Google Scholar n Research which is freely available for redistribution nSubmit your manuscript at submit",
            "",
        ),
        (r"learning ", "learning "),
        (r"training ", "training "),
        (r"into ", "into "),
        (r"conventional ", "conventional "),
        (r"intervention ", "intervention "),
        (r"interventions ", "interventions "),
        (r"information", "information"),
        (r"informed", "informed"),
        (r"nPage \d{1,3} of \d{1,3}", ""),
        (r"nReferences", "References"),
        (r"nMedicine", "Medicine"),
        (r"nare ", "are "),
        (r"nnot ", "not "),
        (r"nour ", "our "),
        (r"nhave ", "have "),
        (r"nsubmit ", "submit "),
        (r"nhealth", "health"),
        (r"nAbstract ", "Abstract "),
        (r"nCorrespondence ", "Correspondence "),
        (r"nCompeting ", "Competing "),
        (r"nAdditional ", "Additional "),
        (r"nKeywords ", "Keywords "),
        (r"ncontrol", "control"),
        (r"ncompared ", "compared "),
        (r"nStatistical ", "Statistical "),
        (r"nstatistical ", "statistical "),
        (r"nPage \d{1,3} of \d{1,3}", ""),
        (r"nbe ", "be "),
        (r"ncells ", "cells "),
        (r"nmore ", "more "),
        (r"significant", "significant"),
        (r"nage", "age"),
        (r"nAge", "Age"),
        (r"damage ", "damage "),
        (r"percentage ", "percentage "),
        (r"nsion", "sion"),
        (r"nli", "li"),
        (r"nbetween ", "between "),
        (r"nexpression ", "expression "),
        (r"nuniversity ", "university "),
        (r"nUniversity ", "University "),
        (r"nmodel ", "model "),
        (r"nanti ", "anti "),
        (r"http ", ""),
        (r"npage", ""),
        (r"nthe", ""),
        (r"ntable", "table"),
        (r"nresults", "results"),
        (r"nbmc", "bmc"),
        (r"nmethods", "methods"),
        (r"nbackground", ""),
        (r"nconclusions", ""),
        (r"nthis", "this"),
        (r"nwe", "we"),
        (r"nafter", "after"),
        (" ment", "ment"),
        (r" tion", "tion"),
        (r"oline", "online"),
        (r" iformed", "informed"),
    ]
    df["text"] = df["text"].astype(str)
    for pattern, replacement in patterns:
        df["text"] = df["text"].str.replace(
            pattern, replacement, regex=True, flags=re.IGNORECASE
        )
    df.drop(df[df["text"].str.len() < 50].index, inplace=True)

    df.to_csv(base_url + "alternative_homeoJ_cleaned.csv", index=False)


def clean_goethe(base_url, input):
    df = pd.read_csv(base_url + input)
    df = df.drop_duplicates()
    df.rename(columns={"category-id": "category_id"}, inplace=True)

    patterns = [
        (r"[ ]{1,2}ment ", "ment "),
        (r"[ ]{1,2}ing ", "ing "),
        (r"\s+ing\s+", "ing "),
        (r"[ ]{1,2}nture ", "ture "),
        (r"http://www\.hindawi\.com", ""),
        (r"www\.hindawi\.com", ""),
        (r"Hindawi", ""),
        (r"http ", ""),
        (r"nThe", "The"),
        (r"nthe", "the"),
        (r"nand", "and"),
        (r"nof", "of"),
        (r"nIn", "In"),
        (r"nto", "to"),
        (r"nwith", "with"),
        (r"nwere", "were"),
        (r"nfor", "for"),
        (r"nwas", "was"),
        (r"ntion", "tion"),
        (r"nResults", "Results"),
        (r"nBMC", "BMC"),
        (r"nMethods", "Methods"),
        (r"nFig", "Fig"),
        (r"nfig ure", "figure"),
        (r"nfigure", "figure"),
        (r"fig ure", "figure"),
        (r"ngroup", "group"),
        (r"nthat", "that"),
        (r"nstudy", "study"),
        (r"ntreatment", "treatment"),
        (r"nhttp", ""),
        (r"nIn", "In"),
        (r"nResults", "Results "),
        (r"nBackground", "Background"),
        (r"nThis", "This"),
        (r"nfrom", "from"),
        (r"nmedicine", "medicine"),
        (r"nment", "ment"),
        (r"nby", "by"),
        (r"nWe", "We"),
        (r"nacupuncture", "acupuncture"),
        (r"npatients", "patients"),
        (r"nTable", "Table"),
        (r"nNo", "No"),
        (r"nConclusions", "Conclusions"),
        (r"nConclusion", "Conclusion"),
        (r"ndata", "data"),
        (r"nafter", "after"),
        (r"neffects", "effects"),
        (r"neffect", "effect"),
        (r"nstudies", "studies"),
        (r"nin", "in"),
        (r"nAll", "All"),
        (r"nreceived", "received"),
        (r"nReceived", "Received"),
        (r"prevention", "prevention"),
        (r"nwhich", "which"),
        (r"nused", "used"),
        (r"nuse ", "use "),
        (r"nusing", "using"),
        (r"nstroke", "stroke"),
        (r"nStroke", "Stroke"),
        (r"nTo", "To"),
        (r"nAuthors", "Authors"),
        (r"nAuthor", "Author"),
        (r"nauthor", "author"),
        (r"nanalysis ", "analysis "),
        (r"nAnalysis ", "Analysis "),
        (r"nDiscussion ", "Discussion "),
        (
            r"nSubmit your next manuscript to BioMed Central and take full advantage of n Convenient online submission n Thorough peer review n No space constraints or color figure charges n Immediate publication on acceptance n Inclusion in PubMed, CAS, Scopus and Google Scholar n Research which is freely available for redistribution nSubmit your manuscript at submit",
            "",
        ),
        (r"learning ", "learning "),
        (r"training ", "training "),
        (r"into ", "into "),
        (r"conventional ", "conventional "),
        (r"intervention ", "intervention "),
        (r"interventions ", "interventions "),
        (r"information", "information"),
        (r"informed", "informed"),
        (r"nPage \d{1,3} of \d{1,3}", ""),
        (r"nReferences", "References"),
        (r"nMedicine", "Medicine"),
        (r"nare ", "are "),
        (r"nnot ", "not "),
        (r"nour ", "our "),
        (r"nhave ", "have "),
        (r"nsubmit ", "submit "),
        (r"nhealth", "health"),
        (r"nAbstract ", "Abstract "),
        (r"nCorrespondence ", "Correspondence "),
        (r"nCompeting ", "Competing "),
        (r"nAdditional ", "Additional "),
        (r"nKeywords ", "Keywords "),
        (r"ncontrol", "control"),
        (r"ncompared ", "compared "),
        (r"nStatistical ", "Statistical "),
        (r"nstatistical ", "statistical "),
        (r"[ ]{1,2}ntive ", "tive "),
        (r"nbe ", "be "),
        (r"ncells ", "cells "),
        (r"nmore ", "more "),
        (r"significant", "significant"),
        (r"nage", "age"),
        (r"nAge", "Age"),
        (r"damage ", "damage "),
        (r"percentage ", "percentage "),
        (r"nsion", "sion"),
        (r"nli", "li"),
        (r"nbetween ", "between "),
        (r"nexpression ", "expression "),
        (r"nuniversity ", "university "),
        (r"nUniversity ", "University "),
        (r"nmodel ", "model "),
        (r"nanti ", "anti "),
        (r"http ", ""),
        (r"ANotated Entity ID \d+ Spans True Boxes True Text", ""),
        (r"Abstract", ""),
        (r"Background", ""),
        (r"npage", ""),
        (r"nthe", ""),
        (r"ntable", "table"),
        (r"nresults", "results"),
        (r"nbmc", "bmc"),
        (r"nmethods", "methods"),
        (r"nbackground", ""),
        (r"nconclusions", ""),
        (r"nthis", "this"),
        (r"nwe", "we"),
        (r"nafter", "after"),
        (" ment", "ment"),
        (r" tion", "tion"),
        (r"oline", "online"),
        (r" iformed", "informed"),
    ]

    df["text"] = df["text"].astype(str)
    for pattern, replacement in patterns:
        df["text"] = df["text"].str.replace(
            pattern, replacement, regex=True, flags=re.IGNORECASE
        )

    df.drop(df[df["text"].str.len() < 50].index, inplace=True)

    df.to_csv(base_url + "alternative_goethe_cleaned.csv", index=False)


def clean_IJRH(base_url, input):
    df = pd.read_csv(base_url + input)
    df = df.drop_duplicates()
    df = df.rename(columns={"category-id": "category_id"})
    df["text"] = df["text"].drop_duplicates()
    print("loaded IJRH data")
    patterns = [
        (r"[ ]{1,2}ment ", "ment "),
        (r"[ ]{1,2}ing ", "ing "),
        (r"\s+ing\s+", "ing "),
        (
            r"\© \d{4} Indian Journal of Research in Homoeopathy \| Published by Wolters Kluwer \- Medknow\d{1,3}",
            "",
        ),
        (r"Indian Journal of Research in Homoeopathy", ""),
        (r"ORIGINAL ARTICLE", ""),
        (r"For reprints contact: reprints@medknow.com", ""),
        ("medknow", ""),
        (r" \[Downloaded free from http\:\/\/www\.ijrh\.org on .+]", ""),
        (r" http ", ""),
        (r" cantly ", "cantly "),
        (r"\s+sion\s+ ", "sion "),
        (r"[ ]{1,2}sion ", "sion "),
        (r"nThe", "The"),
        (r" nthe", " the"),
        (r"nand", "and"),
        (r"nof", "of"),
        (r"nIn", "In"),
        (r"nto", "to"),
        (r"nwith", "with"),
        (r"nwere", "were"),
        (r"nfor", "for"),
        (r"nwas", "was"),
        (r"ntion", "tion"),
        (r"nResults", "Results"),
        (r"nBMC", "BMC"),
        (r"nMethods", "Methods"),
        (r"nFig", "Fig"),
        (r"nfig ure", "figure"),
        (r"nfigure", "figure"),
        (r"fig ure", "figure"),
        (r"ngroup", "group"),
        (r"nthat", "that"),
        (r"nstudy", "study"),
        (r"ntreatment", "treatment"),
        (r"nhttp", " "),
        (r"nBackground", "Background"),
        (r"nThis", "This"),
        (r"nfrom", "from"),
        (r"nmedicine", "medicine"),
        (r"nment", "ment"),
        (r"nby", "by"),
        (r"nWe", "We"),
        (r"nacupuncture", "acupuncture"),
        (r"npatients", "patients"),
        (r"nTable", "Table"),
        (r"nNo", "No"),
        (r"nConclusions", "Conclusions"),
        (r"nConclusion", "Conclusion"),
        (r"ndata", "data"),
        (r"nafter", "after"),
        (r"neffects", "effects"),
        (r"neffect", "effect"),
        (r"nstudies", "studies"),
        (r" nin", " in"),
        (r" nAll", " All"),
        (r" nreceived", " received"),
        (r" nReceived", " Received"),
        (r" prevetion", " prevention"),
        (r" nwhich", " which"),
        (r" nused", " used"),
        (r" nuse ", " use "),
        (r" nusing", " using"),
        (r" nstroke", " stroke"),
        (r" nStroke", " Stroke"),
        (r" nTo", " To"),
        (r" nAuthors", " Authors"),
        (r" nAuthor", " Author"),
        (r" nauthor", " author"),
        (r" nanalysis ", " analysis "),
        (r" nAnalysis ", " Analysis "),
        (r" nDiscussion ", " Discussion "),
        (
            r"nSubmit your next manuscript to BioMed Central and take full advantage of  n  Convenient online submission n  Thorough peer review n  No space constraints or color  gure charges n  Immediate publication on acceptance n  Inclusion in PubMed  CAS  Scopus and Google Scholar n  Research which is freely available for redistribution nSubmit your manuscript at  submit",
            "",
        ),
        (r" learing ", " learning "),
        (r" traiing ", " training "),
        (r" ito ", " into "),
        (r" convetional ", " conventional "),
        (r" intervetion ", " intervention "),
        (r" intervetions ", " interventions "),
        (r" iformation", "information"),
        (r" iformed", "informed"),
        (r"[ ]{1,2}tion", "tion"),
        (r"[ ]{1,2}tions", "tions"),
        (r"nPage \d{1,3} of \d{1,3}", ""),
        (r"nReferences", "References"),
        (r"nMedicine", "Medicine"),
        (r" nare ", " are "),
        (r" nnot ", " not "),
        (r" nour ", " our "),
        (r" nhave ", " have "),
        (
            r"nSubmit your next manuscript to BioMed Central and take full advantage of  n  Convenient online submission n  Thorough peer review n  No space constraints or color  gure charges n  Immediate publication on acceptance n  Inclusion in PubMed  CAS  Scopus and Google Scholar n  Research which is freely available for redistribution nSubmit your manuscript at  submit",
            "",
        ),
        (r" nsubmit ", " submit "),
        (r" nhealth", " health"),
        (r" nAbstract ", " Abstract "),
        (r" nCorrespondence ", "  Correspondence "),
        (r" nCompeting ", " Competing "),
        (r" nAdditional ", " Additional "),
        (r" nKeywords ", " Keywords "),
        (r" ncontrol", " control"),
        (r" ncompared ", " compared "),
        (r" nStatistical ", " Statistical "),
        (r" nstatistical ", " statistical "),
        (r"[ ]{1,2}ntive ", "tive "),
        (r" nbe ", " be "),
        (r" ncells ", " cells "),
        (r" nmore ", " more "),
        (r" signifi  ncant", "significant"),
        (r" nage", " age"),
        (r" nAge", " Age"),
        (r" dam  age ", " damage "),
        (r" percent  age ", " percentage "),
        (r" nsion", "sion"),
        (r" nli", " li"),
        (r" nbetween ", " between "),
        (r" nexpression ", " expression "),
        (r" nuniversity ", " university "),
        (r" nUniversity ", " University "),
        (r" nmodel ", " model "),
        (r" nanti ", " anti "),
        (r"Annotated Entity:\s*ID:\s*\d+\s*Spans:\s*True\s*Boxes:\s*True\s*Text:", ""),
        ("metioned", "mentioned"),
        (r"\[ Abstract ", ""),
        (r"Abstract", ""),
        (r"\[ Objectives:", ""),
        (r"\[ Objective:", ""),
        (r"\[ Background & objectives:", ""),
        (r"Background:", ""),
        (r"Background", ""),
        (r"BackgroundBackground:", ""),
        (r"Introduction:", ""),
        (r"Introductio", ""),
        (r"IntroductioIntroduction:", ""),
        (r"I ntRoductIon", ""),
        (r"ntRoductIon", ""),
        (r"\[ Background & objective:", ""),
        (r"\[ Background:", ""),
        (r"\[ ABSTRACT Background:", ""),
        (r"\[ ABSTRACT", ""),
        (r"AbStRACt", ""),
        (r"\[ Summary:", ""),
        (r"Context:", ""),
        (r"\[ .,", ""),
        (r"\[ .", ""),
        (r"\[ ", ""),
        (r"\[ ,", ""),
        (r"npage", ""),
        (r"nthe", ""),
        (r"ntable", "table"),
        (r"nresults", "results"),
        (r"nbmc", "bmc"),
        (r"nmethods", "methods"),
        (r"nbackground", ""),
        (r"nconclusions", ""),
        (r"nthis", "this"),
        (r"nwe", "we"),
        (r"nafter", "after"),
        (" ment", "ment"),
        (r" tion", "tion"),
        (r"oline", "online"),
    ]

    df["text"] = df["text"].astype(str)
    for pattern, replacement in patterns:
        df["text"] = df["text"].str.replace(pattern, replacement, regex=True)

    df.drop(df[df["text"].str.len() < 50].index, inplace=True)
    df.to_csv(base_url + "alternative_IJRH_cleaned.csv", index=False)


def clean_naturalnews(base_url, input):
    df = pd.read_csv(base_url + input)
    df = df.drop_duplicates()
    print("loaded NaturalNews data")

    patterns = [
        (
            r"Originally published \b(?:January|Febuary|March|April|May|June|July|August|September|October|November|December?) \d{1,2} \d{4}",
            "",
        ),
        (r"NaturalNews.com", ""),
        (r"medknow", ""),
        (
            r"Trends Journal pioneer Gerald Celente joins Mike Adams with breaking news analysis of world events and financial outcomes 20913 Brighteon Broadcast News",
            "",
        ),
        (
            r"May 24 2024 WOKE medical schools spontaneously exploding Boeing airliners and other insanities of a wrecked society 1638 Get ready for HISTORIC ANNOUNCEMENT on Memorial Day 2424 Blinken and Nuland push DANGEROUS war escalation with Russia BEGGING for nuclear retaliation 5806 A whole new take on HISTORY Author Christopher Bjerknes challenges everything you think you know about the history of our world 23422 Brighteon Broadcast News May 23 2024 A new Golden Age era of WEALTH and ABUNDANCE is about to commence for nonwestern nations 1119 Priestess of DEATH Victoria Nuland still trying to start World War III with Russia Home",
            "",
        ),
        (r" naturalnews.com ", ""),
        (r" NaturalNews", ""),
        ("printable article", ""),
        (
            r" NEXT ARTICLE >>Disclaimer.*",
            "",
        ),
        (r"For the full terms of usage of this material visit www.terms.shtml", ""),
        (r" that\\ ", " that "),
        (r" you\\", " you "),
        (r" alzheimer\\ ", " alzheimer "),
        (r" don\\'t ", r" don't "),
        (r" it\\ ", " it"),
        ("Brighteon Broadcast News", ""),
        ("Home Brighteon Prep with Mike", ""),
        (
            "Interviews Audio Books Download Our App About Us FAQs Search Sections Follow Us Podcast Store Subscribe",
            "",
        ),
        ("Home Politics Culture Health", ""),
        ("Medicine Finance", ""),
        ("Economy Prepping", ""),
        (
            "Survival Science Technology Popular Articles Today Week Month Year See More Popular Articles Health Ranger Report",
            "",
        ),
        ("BrightLearn", ""),
        ("Preparing for Invisible Threats  A Reality Check", ""),
        (
            "Western nations are DESTROYING THEMSELVES on purpose as China is winning the AI race",
            "",
        ),
        (
            "Doc Pete Chambers joins Mike Adams to discuss faith and community resilience as wars rage and economic uncertainty strikes home",
            "",
        ),
        (
            "All western nations are actually SATANIC DEATH CULTS masquerading as democracies",
            "",
        ),
        (
            "AI systems will SECRETLY set their own directives to ESCAPE human control",
            "",
        ),
        (
            "Natural Intelligence  Thriving Beyond the System  an interview with Cory Endrulat and Jim Gale",
            "",
        ),
        (
            "Me will WEAPONIZED your genetic data  while Trump nominates PHARMA CLOWN for CDC Home",
            "",
        ),
        ("Conspiracy", ""),
        (
            r"This video is from the\s+\w+\s+\w+\s+channel on Brighteon\.com\. More related stories.*$",
            "",
        ),
        (" 01102024", ""),
    ]

    df["text"] = df["text"].apply(
        lambda x: x.split("Sources for this article include")[0]
    )
    df["text"] = df["text"].apply(
        lambda x: x.split(
            "Take Action Support Natural News by linking to this article from your website."
        )[0]
    )
    # df['text'] = df['text'].apply(lambda x: x.split('printable article')[1])
    df["text"] = df["text"].apply(lambda x: x.split("Permalink to this article")[0])
    df["text"] = df["text"].apply(
        lambda x: x.split("This site is part of the Natural News Network")[0]
    )

    for pattern, replacement in patterns:
        df["text"] = df["text"].str.replace(pattern, replacement, regex=True)

    df.drop(df[df["text"].str.len() < 50].index, inplace=True)
    df.to_csv(base_url + "disinfo_NaturalNews_cleaned.csv", index=False)


def clean_HIN(base_url, input):

    df = pd.read_csv(base_url + input)
    df = df.drop_duplicates()
    print("loaded Health Impact News data")

    df["text"] = df["text"].apply(lambda x: x.split("Read the full article here")[0])
    df["text"] = df["text"].apply(lambda x: x.split("Read the Full Article Here")[0])
    df["text"] = df["text"].apply(lambda x: x.split("Read the Full article Here")[0])
    df["text"] = df["text"].apply(
        lambda x: x.split(
            "Done Please check your email inbox or spam folder for our confirmation email."
        )[0]
    )
    df["text"] = df["text"].apply(lambda x: x.split("We respect your email privacy")[0])
    df["text"] = df["text"].str.replace(r" that\\ ", " that ")
    df["text"] = df["text"].str.replace(r" alzheimer\\ ", " alzheimer ")
    df["text"] = df["text"].str.replace(r" don\\'t ", r" don't ", regex=True)
    df["text"] = df["text"].str.replace(r" it\\ ", " it ")
    df["text"] = df["text"].str.replace(r" you\\ ", " you ")
    df["text"] = df["text"].astype(str)
    df.drop(df[df["text"].str.len() < 50].index, inplace=True)
    df.to_csv(base_url + "disinfo_healthimpactnews_cleaned.csv", index=False)


def clean_mercola(base_url, input):
    df = pd.read_csv(base_url + input)
    df = df.drop_duplicates()
    print("loaded Mercola data")
    patterns = [
        (r" you\\ ", " you "),
        (r" it\\ ", " it "),
        (r" don\'t ", r" don't "),
        (r" alzheimer\ ", " alzheimer "),
        (r" that\\ ", " that "),
        (r"Dr. Mercola encourages you.*", ""),
        ("takecontrol.substack.com", ""),
        (r"Copy linkFacebookEmailNoteOther1Share", ""),
        ("STORY AT-A-GLANCE", ""),
        ("Dr. Joseph Mercola", ""),
        (
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec?) \d{2}, \d{5}Share this post",
            "",
        ),
        (r"takecontrol\.substack\.comCopy linkFacebookEmailNoteOther", ""),
        (r"SubscribeSign inShare this post", ""),
        (r"Dr\. Mercola\'s Censored Library \(Private Membership\)", ""),
        (r"Dr\. Mercola\\\'s Censored Library \(Private Membership\)", ""),
        ("by Dr. Joseph Mercola", ""),
        (r"\[\"", ""),
        (r"\[\'", ""),
    ]

    for pattern, replacement in patterns:
        df["text"] = df["text"].str.replace(pattern, replacement, regex=True)

    def extract_text_5(x):
        try:
            return re.split(r"Continue readingSign in", x)[1]
        except Exception:
            return x

    df["text"] = df["text"].apply(extract_text_5)

    df.drop(df[df["text"].str.len() < 50].index, inplace=True)
    df.to_csv(base_url + "disinfo_mercola_cleaned.csv", index=False)


def clean_HealthDOTNews(base_url, input):
    df = pd.read_csv(base_url + input)
    df = df.drop_duplicates()
    print("loaded Health DOT News data")
    patterns = [
        (
            "Your Name Your email Message  or Cancel                      SCIENCE FOOD HEALTH MEDICINE POLLUTION CANCER CLIMATE",
            "",
        ),
        (r" that\\ ", " that "),
        (r" alzheimer\\ ", " alzheimer "),
        (r" don\\'t ", r" don't "),
        (r" it\\ ", " it "),
        (r" you\\ ", " you "),
    ]

    df["text"] = df["text"].astype(str)

    df["text"] = df["text"].str.strip()
    for pattern, replacement in patterns:
        df["text"] = df["text"].str.replace(pattern, replacement, regex=True)

    df.drop(df[df["text"].str.len() < 50].index, inplace=True)
    df.to_csv(base_url + "disinfo_healthDOTnews_cleaned.csv", index=False)


def clean_infowars(base_url, input):
    df = pd.read_csv(base_url + input)
    df = df.drop_duplicates()
    print("loaded InfoWars data")
    df["text"] = df["text"].dropna()
    # df['text'] = df['text'].str.strip('NaN')
    df["text"] = df["text"].str.replace(
        "Banned.VideoInfowars StoreNews WarsInfowars LifeSearchSearch ResultsNo Search Results FoundLIVESearchSearch ResultsNo Search Results Found Explore HomeNewsPodcastsBreaking NewsSocial Watch Live Infowars NetworkThe Alex Jones ShowThe War Room with Owen ShroyerThe American Journal More Banned.VideoInfowars StoreArchiveRSSDownload Our AppTerms of ServiceDMCAAdvertise with usAffiliatesMedia",
        "",
    )
    df["text"] = df["text"].str.replace("InquiriesAbout", "")
    df["text"] = df["text"].str.replace(
        "SharePostTweetMessageEmailLIVESharePostTweetMessageEmailLIVEposted Invalid date agoView More From Terms of ServiceDMCAAdvertise with usAffiliatesMedia InquiriesAbout",
        "",
    )
    df["text"] = df["text"].str.replace(
        "SharePostTweetMessageEmailLIVESharePostTweetMessageEmailLIVEposted Invalid date agoView More From Terms of ServiceDMCAAdvertise with usAffiliatesMedia",
        "",
    )
    df["text"] = df["text"].str.replace(
        "DMCAAdvertise with usAffiliatesMedia", ""
    )  ### attention: we first cleaned and filtered for topics afterwards

    df_1 = pd.read_csv(
        base_url + "before_cleaning/FSoLS-25-v5-_fulltext/disinfo_infowars.csv",
        on_bad_lines="skip",
        sep=None,
        engine="python",
        encoding="utf-8",
    )
    df_2 = pd.read_csv(
        base_url
        + "before_cleaning/FSoLS-25-v5-_fulltext/testset/disinfo_infowars-3_2025-03-28.csv",
        on_bad_lines="skip",
        sep=None,
        engine="python",
        encoding="utf-8",
    )
    df = pd.concat([df_1, df_2], axis=0, ignore_index=True).drop_duplicates()
    # df = pd.read_csv(base_url + 'before_cleaning/FSoLS-25-v5-_fulltext/disinfo_infowars.csv', on_bad_lines='skip')
    df["text"] = df["text"].str.replace("Terms of Service", "")
    df["text"] = df["text"].str.replace("| Infowars.com", "")
    df["text"] = df["text"].str.replace(r" that\\ ", " that ")
    df["text"] = df["text"].str.replace(r" alzheimer\\ ", " alzheimer ")
    df["text"] = df["text"].str.replace(r" don\\'t ", r" don't ", regex=True)
    df["text"] = df["text"].str.replace(r" it\\ ", " it ")
    df["text"] = df["text"].str.replace(r" you\\ ", " you ")
    df["text"] = df["text"].astype(str)
    df.drop(df[df["text"].str.len() < 50].index, inplace=True)
    print(df["text"].head())
    df = df.rename(columns={"data_source": "data-source"})
    df.to_csv(base_url + "disinfo_infowars_cleaned.csv", index=False)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "base_url"
    )  # Path to the base_url containing input files also for output files
    argparser.add_argument("PMC_input")  # Path to PMC file
    argparser.add_argument("WebMD_input")  # Path to WebMD file
    argparser.add_argument("HHP_input")  # Path to Harvard Health Publishing file
    argparser.add_argument("MH_input")  # Path to Men's Health file
    argparser.add_argument("WH_input")  # Path to Womens Health file
    argparser.add_argument("MP_input")  # Path to Medline Plus file
    argparser.add_argument("JEBIM_input")  # Path to JEBIM file
    argparser.add_argument("CMT_input")  # Path to CMT file
    argparser.add_argument("homeoJournal_input")  # Path to Homeo Journal file
    argparser.add_argument("Goethe_input")  # Path to Goethe file
    argparser.add_argument("IJRH_input")  # Path to IJRH file
    argparser.add_argument("NN_input")  # Path to Natural News file
    argparser.add_argument("HIN_input")  # Path to HIN file
    argparser.add_argument("Mercola_input")  # Path to Mercola file
    argparser.add_argument("HealthDOTNews_input")  # Path to Health.News file
    argparser.add_argument("Infowars_input")  # Path to Infowars file
    args = argparser.parse_args()

    base_url = args.base_url
    col_list = ["category_id", "tags", "data-source", "text"]

    # sientific
    clean_pmc(base_url, args.PMC_input)

    # vernacular
    clean_webmd(base_url, args.WebMD_input)
    clean_harvardHealthPublishing(base_url, args.HHP_input)
    clean_MH(base_url, args.MH_input)
    clean_WH(base_url, args.WH_input)
    clean_MedlinePlus(base_url, col_list, args.MP_input)

    # alternative scientific
    clean_JEBIM(base_url, col_list, args.JEBIM_input)
    clean_CMT(base_url, args.CMT_input)
    clean_homeoJ(base_url, args.homeoJournal_input)
    clean_goethe(base_url, args.Goethe_input)
    clean_IJRH(base_url, args.IJRH_input)

    # disinfo
    clean_naturalnews(base_url, args.NN_input)
    clean_HIN(base_url, args.HIN_input)
    clean_mercola(base_url, args.Mercola_input)
    clean_HealthDOTNews(base_url, args.HealthDOTNews_input)
    clean_infowars(base_url, args.Infowars_input)
    print("done")


if __name__ == "__main__":
    main()
