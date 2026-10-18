from datetime import date
import pytest
from jack_flavours.data import parse_text, normalize, audit


def menu(body, day='October 17th, 2026', footer='Single Scoop £2.90'):
    return '\ufeffGelato\n'+day+'\n-\n'+body+'\n'+footer+'\nPlease note that flavours may sell out.'


def test_boundaries_and_annotations():
    text=menu('Vanilla\nMango (vegan) (+£2)\nHot Chocolate!\nSee inside for selection')
    status,_,items,_=parse_text(text,date(2026,10,17))
    assert status=='valid'
    assert [x['canonical_name'] for x in items]==['vanilla','mango']
    assert items[1]['raw_name']=='Mango (vegan) (+£2)'
    assert 'vegan' in items[1]['annotations']


@pytest.mark.parametrize('text,reason',[(menu('Vanilla',day='October 16th, 2026'),'menu_date_mismatch'),('Closed for refurbishment','closure_notice'),(menu(''),'empty_menu'),(menu('Vanilla\nvanilla'),'duplicate_flavour'),('October 17th, 2026\nVanilla\n','missing_footer_boundary'),(menu('Shop is open tomorrow'),'unexpected_menu_line')])
def test_suspect_is_never_absence(text,reason):
    status,_,items,warnings=parse_text(text,date(2026,10,17))
    assert status=='suspect' and items==[] and reason in warnings


def test_holiday_and_no_separator():
    assert parse_text(menu('Vanilla',day='Halloween, 2024'),date(2024,10,31))[0]=='valid'
    assert parse_text(menu('Vanilla').replace('\n-\n','\n'),date(2026,10,17))[0]=='valid'


def test_normalization_does_not_merge_flavours():
    assert normalize(' Coconut  Ube (vegan) ')== 'coconut ube'
    assert normalize('Coconut & Ube') != normalize('Coconut Ube')


def test_audit_missing_and_filename_variants(tmp_path):
    raw=tmp_path/'source'/'raw';raw.mkdir(parents=True)
    (raw/'26_10_17__all_saints.txt').write_text(menu('Vanilla'))
    (raw/'26_10_19_benet_street.txt').write_text(menu('Chocolate',day='October 19th, 2026'))
    result=audit(raw.parent,tmp_path/'out',tmp_path/'report.md',{'source_commit':'test','shops':['all_saints','benet_street']},verify_source=False)
    assert result['source_files']==2 and result['status_counts']=={'missing':4,'valid':2}
    assert result['appearances']==2
    (raw/'26_10_17_all_saints.txt').write_text(menu('Vanilla'))
    with pytest.raises(ValueError,match='duplicate'):
        audit(raw.parent,tmp_path/'out',tmp_path/'report.md',{'source_commit':'test','shops':['all_saints','benet_street']},verify_source=False)
