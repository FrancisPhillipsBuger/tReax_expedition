#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Public API for the treax_expedition package: XYZ I/O, geometry helpers,
the GFN2-xTB, ORCA and free EMT calculator setups, and the SWS Monte-Carlo
samplers for one or two structures.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Public re-exports
from .io_xyz import (EvaluationLog, listReactions, prepareOutputFolder,
                     reactionName, readMeta, readXYZFile, resolveFolder,
                     resolveInput, writeXYZFile)
from .geometry import (EndpointAxis, MaybeCenter, MaybeOrient, NeedsFrameRemoval,
                       calcAlignedRMSDBetweenStructures,
                       calcRMSDBetweenStructures, centerMolecule,
                       displaceSWSStructure, orientStructureWrtAnother)
from .calculators import (getEMTCalculator, setCalculatorToStructure,
                          wrapGetPotentialEnergy)
from .orca import getORCACalculator
from .xtb import getXTBCalculator, normaliseMethod
from .sampler import (RunStats, doSingleStructureSWSMC, doTwoStructureSWSMC,
                      mcSingleStructureStep, mcTwoStructuresStep)

__all__ = [
    'listReactions',
    'prepareOutputFolder',
    'reactionName',
    'readMeta',
    'readXYZFile',
    'resolveFolder',
    'resolveInput',
    'EvaluationLog',
    'writeXYZFile',
    'calcRMSDBetweenStructures',
    'calcAlignedRMSDBetweenStructures',
    'orientStructureWrtAnother',
    'centerMolecule',
    'NeedsFrameRemoval',
    'EndpointAxis',
    'MaybeCenter',
    'MaybeOrient',
    'displaceSWSStructure',
    'wrapGetPotentialEnergy',
    'getXTBCalculator',
    'getORCACalculator',
    'normaliseMethod',
    'getEMTCalculator',
    'setCalculatorToStructure',
    'RunStats',
    'mcSingleStructureStep',
    'mcTwoStructuresStep',
    'doSingleStructureSWSMC',
    'doTwoStructureSWSMC',
]
