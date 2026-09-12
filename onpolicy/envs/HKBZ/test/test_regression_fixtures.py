"""Fixture completeness and provenance are part of clean-checkout portability."""

import json
import unittest

from onpolicy.envs.HKBZ.test.regression_fixtures import FIXTURE_ROOT, case_dir


class RegressionFixtureTest(unittest.TestCase):
    def test_all_pinned_cases_are_local_train_fixtures_with_exact_bytes(self):
        index = json.loads((FIXTURE_ROOT / 'provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(set(index['cases']), {'train_case_0012', 'train_case_0046'})
        for name, record in index['cases'].items():
            with self.subTest(fixture=name):
                path = case_dir(name)
                self.assertEqual(path.parent, FIXTURE_ROOT.resolve())
                self.assertEqual(record['original_split'], 'train')
                self.assertEqual({p.name for p in path.iterdir()}, {
                    'job.json', 'flights.json', 'fixed_resources.json',
                    'mobile_resources.json', 'sites.json', 'metadata.json',
                })
                self.assertEqual(json.loads((path / 'metadata.json').read_text())['split'], 'train')

    def test_unknown_or_traversing_fixture_is_not_an_external_fallback(self):
        for name in ('../dataset', 'case_0001', '/tmp/unknown'):
            with self.subTest(fixture=name), self.assertRaises(ValueError):
                case_dir(name)


if __name__ == '__main__':
    unittest.main()
