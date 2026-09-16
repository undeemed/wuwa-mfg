# SPDX-License-Identifier: Apache-2.0
"""Fetch checksum-pinned official DirectML build dependencies into a local directory."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

VERSION='1.15.4'
PACKAGE_SHA='4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9'
HEADER_COMMIT='8700779fe7a09ea7a007cf3d7ab4293c78e41017'
HEADER_SHA='21eb6af105498026ed91a6ea6deb6850c19cbf5302ec3001aee64f6ff5ec4ccd'
EXTRACT=('include/DirectML.h','bin/x64-win/DirectML.dll','bin/x64-win/DirectML.lib','LICENSE.txt')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    url=f'https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/{VERSION}/microsoft.ai.directml.{VERSION}.nupkg'
    package=urllib.request.urlopen(url,timeout=120).read()
    if hashlib.sha256(package).hexdigest()!=PACKAGE_SHA:raise ValueError('DirectML package checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        # Exact Windows x64 paths: never select the similarly named Xbox files.
        for name in EXTRACT:(a.output/Path(name).name).write_bytes(archive.read(name))
    header=urllib.request.urlopen(f'https://raw.githubusercontent.com/microsoft/DirectML/{HEADER_COMMIT}/Libraries/DirectMLX.h',timeout=60).read()
    if hashlib.sha256(header).hexdigest()!=HEADER_SHA:raise ValueError('DirectMLX checksum mismatch')
    (a.output/'DirectMLX.h').write_bytes(header)
    (a.output/'provenance.json').write_text(json.dumps({'version':VERSION,'package_sha256':PACKAGE_SHA,
        'header_commit':HEADER_COMMIT,'header_sha256':HEADER_SHA,'extracted':list(EXTRACT)},indent=2)+'\n')
    print('Official DirectML dependencies verified.')


if __name__=='__main__':main()
