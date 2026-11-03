import pytest

from djimaging.tables.core.userinfo import UserInfoTemplate
from tests.autorois.test_roi_canvas_channels import FakeChannelTable


class FakeUserInfo:
    """Stand-in with the UserInfoTemplate methods and a single user row."""
    get_ch_names = UserInfoTemplate.get_ch_names
    get_stack_names = UserInfoTemplate.get_stack_names
    get_functional_ch_names = UserInfoTemplate.get_functional_ch_names

    def __init__(self, use_channels, channel_table=None, row=None):
        self._use_channels = use_channels
        self.channel_table = lambda: channel_table
        self.row = row

    def __and__(self, key):
        return self

    def fetch1(self, *attrs):
        values = tuple(self.row[attr] for attr in attrs)
        return values[0] if len(values) == 1 else values


def user_info(**kwargs):
    return type('UserInfo', (UserInfoTemplate,), kwargs)()


def test_definition_contains_stack_names_by_default():
    definition = user_info().definition
    assert "data_stack_name='wDataCh0'" in definition and "alt_stack_name='wDataCh1'" in definition


def test_definition_without_stack_names_with_channels():
    definition = user_info(_use_channels=True).definition
    assert 'data_stack_name' not in definition and 'alt_stack_name' not in definition
    assert 'raw_data_dir' in definition


def test_channel_table_required_with_channels():
    with pytest.raises(NotImplementedError):
        user_info(_use_channels=True).channel_table


def test_stack_names_from_user_info():
    table = FakeUserInfo(False, row=dict(data_stack_name='main', alt_stack_name='alt'))
    assert table.get_stack_names({}) == ('main', 'alt')
    assert table.get_ch_names({}) == ['main', 'alt']

    table = FakeUserInfo(False, row=dict(data_stack_name='main', alt_stack_name='main'))
    assert table.get_ch_names({}) == ['main']


def test_stack_names_from_channel_table():
    channels = FakeChannelTable()
    channels.add_channels([('wDataCh0', 0), ('wDataCh1', 1), ('wDataCh2', 2)], roi_channel='wDataCh1')
    table = FakeUserInfo(True, channels)
    assert table.get_ch_names() == ['wDataCh1', 'wDataCh0', 'wDataCh2']
    assert table.get_stack_names() == ('wDataCh1', 'wDataCh0')


def test_single_channel_has_no_alt_stack():
    channels = FakeChannelTable()
    channels.add_channels([('wDataCh0', 0)])
    assert FakeUserInfo(True, channels).get_stack_names() == ('wDataCh0', None)


def test_functional_ch_names():
    channels = FakeChannelTable()
    channels.add_channels([('wDataCh0', 0), ('wDataCh1', 1), ('wDataCh2', 2)], functional=['wDataCh2', 'wDataCh0'])
    assert FakeUserInfo(True, channels).get_functional_ch_names() == ['wDataCh0', 'wDataCh2']

    table = FakeUserInfo(False, row=dict(data_stack_name='main', alt_stack_name='alt'))
    assert table.get_functional_ch_names({}) == ['main']  # The alternative stack is not functional
