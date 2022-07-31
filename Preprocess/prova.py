
import pandas as pd
import re

import time
from time import time

import sys

testo = r"ns1__srmReleaseFiles : Request: Release files. IP: 2001:1458:d00:9::100:130. Client DN: /DC=ch/DC=cern/OU=Organic Units/OU=Users/CN=ddmadmin/CN=531497/CN=Robot: ATLAS Data Management. surl(s): srm://storm-fe.cr.cnaf.infn.it/atlas/ ns1__srmReleaseFiles : Request: Release files. IP: 2001:1458:201:e4::100:574. Client DN: /DC=ch/DC=cern/OU=Organic Units/OU=Users/CN=ddmadmin/CN=531497/CN=Robot: ATLAS Data Management. surl(s): srm://storm-fe.cr.cnaf.infn.it/atlas/atlasdatadisk/rucio/tests/c1/d2/step14.77370.9173.recon.ESD.93747.49296. token: b9131460-2893-4891-8a0d-797be23db172"

URL = re.compile('srm:\/\/storm-fe.cr.cnaf.infn.it\/[a-zA-Z0-9_/.]+')
testoMask = re.sub(URL, '<URL>', testo)

print(testoMask)

