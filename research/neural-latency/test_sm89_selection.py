# SPDX-License-Identifier: MIT
"""Compile the actual demo-only selector against synthetic malformed fatbins.

Run in a VS x64 developer shell. No NVIDIA payload is read or executed.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--patch', type=Path, default=Path(__file__).with_name('optiscaler-demo-launch-contract.patch'))
args = parser.parse_args()
patch = args.patch.read_text()
section = patch.split('+++ b/OptiScaler/dlssnr/DlssNr_DemoKernelProbe.h\n', 1)[1]
section = section.split('\ndiff --git', 1)[0]
header = '\n'.join(line[1:] for line in section.splitlines()
                   if line.startswith('+') and not line.startswith('+++'))
selector = header.split('inline std::vector<unsigned char> SelectSm89', 1)[1]
selector = 'inline std::vector<unsigned char> SelectSm89' + selector.split('inline NvAPI_Status', 1)[0]
prefix = r'''
#include <vector>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>
using NvU32=std::uint32_t;
bool forceSm89=true;
'''
harness = r'''
template<class T> void put(std::vector<unsigned char>& b,size_t at,T value)
{ memcpy(b.data()+at,&value,sizeof(value)); }
std::vector<unsigned char> blob(std::initializer_list<unsigned> architectures)
{
    std::vector<unsigned char> b(16+architectures.size()*72,0);
    put(b,0,0xba55ed50u);put(b,4,(unsigned short)1);put(b,6,(unsigned short)16);
    put(b,8,(unsigned long long)(b.size()-16));size_t at=16;
    for(auto sm:architectures) {
        put(b,at,(unsigned short)2);put(b,at+2,(unsigned short)257);
        put(b,at+4,64u);put(b,at+8,8ull);put(b,at+28,sm);
        for(size_t i=at+64;i<at+72;++i)b[i]=(unsigned char)sm;
        at+=72;
    }
    return b;
}
auto select(const std::vector<unsigned char>& b) {return SelectSm89(b.data(),(NvU32)b.size());}
int main()
{
    auto b=blob({75,86,89,120});auto expected=blob({89});
    assert(select(b)==expected);
    auto padded=b;padded.resize(b.size()+256);assert(select(padded)==expected);
    padded.back()=1;assert(select(padded).empty());
    forceSm89=false;assert(select(b).empty());forceSm89=true;
    assert(SelectSm89(nullptr,1024).empty());
    for(unsigned n=0;n<16;++n)assert(SelectSm89(b.data(),n).empty());
    auto bad=b;put(bad,0,0u);assert(select(bad).empty());
    bad=b;put(bad,4,(unsigned short)2);assert(select(bad).empty());
    bad=b;put(bad,6,(unsigned short)8);assert(select(bad).empty());
    bad=b;put(bad,8,~0ull);assert(select(bad).empty());
    bad=b;put(bad,20,31u);assert(select(bad).empty());
    bad=b;put(bad,20,513u);assert(select(bad).empty());
    bad=b;put(bad,24,~0ull);assert(select(bad).empty());
    bad=b;bad.pop_back();assert(select(bad).empty());
    assert(select(blob({75,86,120})).empty());
    assert(select(blob({89,89})).empty());
    bad=blob({89});put(bad,16,(unsigned short)1);assert(select(bad).empty());
    bad=blob({89});put(bad,8,73ull);bad.push_back(0);assert(select(bad).empty());
    std::cout << "SM89 selector: exact preservation, gating, padding and malformed-input checks passed\n";
}
'''
with tempfile.TemporaryDirectory(prefix='nr-selector-test-') as folder:
    root = Path(folder)
    cpp = root / 'selector.cpp'
    cpp.write_text(prefix + selector + harness)
    compile_result = subprocess.run(['cl', '/nologo', '/EHsc', '/std:c++17',
                                     str(cpp), '/Fe:selector.exe'], cwd=root,
                                    capture_output=True, text=True, timeout=60)
    if compile_result.returncode:
        raise SystemExit(compile_result.stdout + compile_result.stderr)
    subprocess.run([str(root / 'selector.exe')], cwd=root, check=True, timeout=15)
