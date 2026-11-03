"""Expand only the hash-listed historical inputs to a new scratch directory.
No training, model loading, query evaluation, download, package install or
change to archived inputs is performed. A nonempty unrelated directory is refused.
"""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path, PurePosixPath

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    if out==ROOT or out in ROOT.parents or out==HERE or out in HERE.parents:
        raise ValueError('OUTPUT_CONTAINS_RELEASE_FILES')
    manifest=list(csv.DictReader(io.StringIO((HERE/'INPUT_MANIFEST.csv').read_text(encoding='utf-8'))))
    hashes={r['path']:r for r in manifest}
    with zipfile.ZipFile(HERE/'FROZEN_INPUTS.zip') as z:
        if set(z.namelist())!=set(hashes):raise ValueError('ARCHIVE_MEMBER_SET')
        for name,r in hashes.items():
            rel=PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts or '\\' in name:
                raise ValueError('UNSAFE_MEMBER_PATH')
            blob=z.read(name)
            if len(blob)!=int(r['bytes']) or hashlib.sha256(blob).hexdigest()!=r['sha256']:
                raise ValueError('SOURCE_HASH:'+name)
        if out.exists():
            actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
            if actual-set(hashes):raise ValueError('OUTPUT_HAS_UNRELATED_FILES')
            for name in actual:
                p=out/name
                if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[name]['sha256']:
                    raise ValueError('OUTPUT_CONFLICT:'+name)
        for name,r in hashes.items():
            p=out/name
            # Refuse an existing symlink anywhere in the output path.
            cur=p
            while cur!=out.parent:
                if cur.is_symlink():raise ValueError('OUTPUT_SYMLINK')
                cur=cur.parent
            if not p.exists():
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    print('PREPARED_FROZEN_INPUT_FILES='+str(len(hashes)))
    print('TRAINING_QUERY_DOWNLOAD_INSTALL=NOT_RUN')
    print('OUTPUT='+str(out))
if __name__=='__main__':main()
