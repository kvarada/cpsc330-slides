import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('pdfs', Path(__file__).with_name('slide_pdfs.py'))
pdfs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdfs)


class CacheTests(unittest.TestCase):
    def test_publication_and_invalidation(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                def write(name, text):
                    p = Path(name)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(text)
                write('.github/workflows/publish.yml', 'workflow')
                for n in (1, 2):
                    write(f'website/lecture-0{n}.qmd', f'---\npublish-pdf: true\n---\n{{{{< revealjs "slides/slides-0{n}.html" >}}}}')
                    write(f'website/_site/slides/slides-0{n}.html', f'deck {n}')
                def pending():
                    pdfs.plan()
                    return json.loads(pdfs.PLAN.read_text())
                jobs = pending()
                self.assertEqual(len(jobs), 2)
                for job in jobs:
                    name = Path(job['stem']).name
                    write(f'.pdf-cache/{name}.pdf', '%PDF-fixture')
                    write(f'.pdf-cache/{name}.json', json.dumps(job['key']))
                self.assertEqual(pending(), [])
                write('website/_site/index.html', 'homepage edit')
                self.assertEqual(pending(), [])
                write('website/_site/slides/slides-01_files/plot.png', 'changed plot')
                self.assertEqual([j['stem'] for j in pending()], ['slides/slides-01'])
                write('website/_site/slides/img/shared.png', 'shared image')
                self.assertEqual(len(pending()), 2)
                page = Path('website/lecture-01.qmd')
                page.write_text(page.read_text().replace('true', 'false'))
                self.assertEqual([j['stem'] for j in pending()], ['slides/slides-02'])
                self.assertFalse(Path('website/_site/slides/slides-01.pdf').exists())
            finally:
                os.chdir(original)


if __name__ == '__main__':
    unittest.main()
