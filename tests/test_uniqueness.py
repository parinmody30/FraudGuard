"""Uniqueness of ids across transactions, merchants, and devices."""

from __future__ import annotations

from fraudlab.generators import DeviceGenerator, MerchantGenerator, PopulationGenerator


def test_transaction_ids_unique(small_dataset):
    transactions, _, _ = small_dataset
    ids = [t.transaction_id for t in transactions]
    assert len(ids) == len(set(ids))


def test_merchant_ids_unique(small_config):
    merchants = MerchantGenerator(small_config).generate()
    ids = [m.merchant_id for m in merchants]
    assert len(ids) == len(set(ids))


def test_device_ids_unique(small_config):
    users, _ = PopulationGenerator(small_config).generate()
    devices = DeviceGenerator(small_config).generate(users)
    ids = [d.device_id for pool in devices.values() for d in pool]
    assert len(ids) == len(set(ids))


def test_user_ids_unique(small_config):
    users, accounts = PopulationGenerator(small_config).generate()
    user_ids = [u.user_id for u in users]
    assert len(user_ids) == len(set(user_ids)) == small_config.num_users
    assert set(accounts.keys()) == set(user_ids)
