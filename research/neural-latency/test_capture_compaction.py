# SPDX-License-Identifier: Apache-2.0
"""Verify transparent compression on a fresh copy; original captures stay untouched."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from compact_teacher_capture import compact_capture,capture_files,digest
from compare_output_grade import read_capture


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.capture.resolve(strict=True);output=args.output.resolve()
    repo=Path(__file__).resolve().parents[2]
    assert not output.exists() and not output.is_relative_to(repo)
    original=capture_files(source,source.parent)
    original_metadata={f'frame-{i}.json':digest(source/f'frame-{i}.json') for i in range(4)}
    output.mkdir(parents=True);copied=output/'capture';copied.mkdir()
    for path,_ in original:shutil.copyfile(path,copied/path.name)
    for name in original_metadata:shutil.copyfile(source/name,copied/name)
    before=read_capture(source)
    report=compact_capture(copied,output)
    after=read_capture(copied)
    assert before[0]==after[0] and before[2]==after[2]
    assert all(np.array_equal(before[1][role],after[1][role]) for role in ('color','output'))
    for path,row in original:
        assert digest(path)==row['sha256'] and path.stat().st_size==row['logical_bytes']
    assert all(digest(source/name)==value==digest(copied/name) for name,value in original_metadata.items())
    rejected=[]
    for label,root in [('root-equals-capture',copied),('outside-root',output/'missing-root')]:
        try:capture_files(copied,root)
        except (ValueError,FileNotFoundError):rejected.append(label)
        else:raise AssertionError(label)
    result={'complete':True,'target_achieved':False,'quality_gate_passed':False,'compression':report,
            'original_capture_and_metadata_unchanged':True,'existing_reader_pixels_exact':True,
            'metadata_unchanged':True,'rejected_roots':rejected}
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='compression'}|{k:report[k] for k in ['logical_bytes','allocated_before','allocated_after','disk_free_delta']},indent=2))


if __name__=='__main__':main()
