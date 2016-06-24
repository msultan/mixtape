import mdtraj as md
import numpy as np
import warnings

from msmbuilder.example_datasets import fetch_fs_peptide
from msmbuilder.featurizer import ContactFeaturizer
from msmbuilder.featurizer import BinaryContactFeaturizer
from msmbuilder.featurizer import LogisticContactFeaturizer

def test_contacts():
    dataset = fetch_fs_peptide()
    trajectories = dataset["trajectories"]
    contactfeaturizer = ContactFeaturizer()
    contacts = contactfeaturizer.transform([trajectories[0]])

    assert contacts[0].shape[1] == 171


def test_binaries():
    dataset = fetch_fs_peptide()
    trajectories = dataset["trajectories"]
    binarycontactfeaturizer = BinaryContactFeaturizer()
    binaries = binarycontactfeaturizer.transform([trajectories[0]])

    assert binaries[0].shape[1] == 171
    assert np.sum(binaries[0]) <= binaries[0].shape[0]*binaries[0].shape[1]


def test_binaries_inf_cutoff():
    dataset = fetch_fs_peptide()
    trajectories = dataset["trajectories"]
    binarycontactfeaturizer = BinaryContactFeaturizer(cutoff=1e10)
    binaries = binarycontactfeaturizer.transform([trajectories[0]])

    assert binaries[0].shape[1] == 171
    assert np.sum(binaries[0]) == binaries[0].shape[0]*binaries[0].shape[1]


def test_binaries_zero_cutoff():
    dataset = fetch_fs_peptide()
    trajectories = dataset["trajectories"]
    binarycontactfeaturizer = BinaryContactFeaturizer(cutoff=0)
    binaries = binarycontactfeaturizer.transform([trajectories[0]])

    assert binaries[0].shape[1] == 171
    assert np.sum(binaries[0]) == 0


def test_logistics():
    dataset = fetch_fs_peptide()
    trajectories = dataset["trajectories"]
    logisticcontactfeaturizer = LogisticContactFeaturizer()
    logistics = logisticcontactfeaturizer.transform([trajectories[0]])

    assert logistics[0].shape[1] == 171
    assert np.amax(logistics[0]) < 1.0
    assert np.amin(logistics[0]) > 0.0


def test_distance_to_logistic():
    dataset = fetch_fs_peptide()
    trajectories = dataset["trajectories"]
    steepness = np.absolute(10*np.random.randn())
    center = np.absolute(np.random.randn())
    contactfeaturizer = ContactFeaturizer()
    contacts = contactfeaturizer.transform([trajectories[0]])
    logisticcontactfeaturizer = LogisticContactFeaturizer(center=center, steepness=steepness)
    logistics = logisticcontactfeaturizer.transform([trajectories[0]])

    for n in range(0,10):
        i = np.random.randint(0, contacts[0].shape[0] - 1)
        j = np.random.randint(0, contacts[0].shape[1] - 1)
    
        x = contacts[0][i][j]
        y = logistics[0][i][j]

        if(x > center):
            assert y < 0.5
        if(x < center):
            assert y > 0.5
        n = n + 1


def test_binary_to_logistics():
    dataset = fetch_fs_peptide()
    trajectories = dataset["trajectories"]
    steepness = np.absolute(10*np.random.randn())
    center = np.absolute(np.random.randn())
    binarycontactfeaturizer = BinaryContactFeaturizer(cutoff=center)
    binaries = binarycontactfeaturizer.transform([trajectories[0]])
    logisticcontactfeaturizer = LogisticContactFeaturizer(center=center, steepness=steepness)
    logistics = logisticcontactfeaturizer.transform([trajectories[0]])

    # This checks that no distances that are larger than the center are logistically
    # transformed such that they are less than 1/2
    np.testing.assert_array_almost_equal(binaries[0], logistics[0] >  0.5 )


