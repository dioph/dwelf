from .. import Modeler
from astropy.io import ascii

kappa2003 = ascii.read('dwelf/data/kappaCeti2003.csv')
kep17 = ascii.read('dwelf/data/kepler17q01.csv')


def test_file_format_kappa2003():
    assert kappa2003.colnames == ['time', 'flux', 'flux_err']
    assert kappa2003['flux'].size == kappa2003['time'].size
    assert kappa2003['time'].size == 419


def test_file_format_kepler17q01():
    assert kep17.colnames == ['time', 'flux', 'flux_err']
    assert kep17['flux'].size == kep17['time'].size
    assert kep17['time'].size == 1626


def test_modeler_class_constructor():
    model = Modeler()
    assert model.x.size == 500
