#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Two-dimensional model potentials sharing one interface: the proton-transfer
double well coupled to a harmonic bath, and the scaled Mueller-Brown and
two-channel potentials of committor_2.0. A call on one point returns cm-1.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import scipy.constants as spc
import scipy.optimize as spo

from constants.units import ConvCoeffForce, PECoeff, wvn2eV

#%% 1. Proton-transfer model parameters
Vpt = 2000.     # proton transfer barrier in cm-1
X0 = 0.5        # position of the minimum in angstroms
vbath = 1000.   # harmonic oscillator frequency in cm-1
alpha = 0.7     # coupling between the bath and proton transfer
mass = 1.0      # mass of the particle in a.m.u.s

#%% 2. Proton-transfer surface
def PotentialEnergy(R, Vpt=Vpt, x0=X0, v=vbath, alpha=alpha, mass=mass):

    """

    Energy of the coupled double-well / bath model.

    :param1 R:     coordinates (x, y) in angstroms; x and y may be arrays.
    :param2 Vpt:   proton transfer barrier in cm-1.
    :param3 x0:    position of the minimum in angstroms.
    :param4 v:     harmonic oscillator frequency in cm-1.
    :param5 alpha: coupling between the bath and proton transfer.
    :param6 mass:  mass of the particle in a.m.u.s.

    :return: E: potential energy in cm-1.

    """

    x = R[0]
    y = R[1]
    E = Vpt * (x - x0)**2 * (x + x0)**2/x0**4
    # force constant in cm-1 / angstroem**2
    f = v * (mass * ConvCoeffForce * v)

    E += 0.5 * f * y**2 - f * alpha * x * y

    return E

def PotentialGradient(R, Vpt=Vpt, x0=X0, v=vbath, alpha=alpha, mass=mass):

    """

    Analytic gradient of the coupled double-well / bath model.

    :param1 R:     coordinates (x, y) in angstroms; x and y may be arrays.
    :param2 Vpt:   proton transfer barrier in cm-1.
    :param3 x0:    position of the minimum in angstroms.
    :param4 v:     harmonic oscillator frequency in cm-1.
    :param5 alpha: coupling between the bath and proton transfer.
    :param6 mass:  mass of the particle in a.m.u.s.

    :return: G: gradient in cm-1 per angstroem, (dx, dy).

    """

    x = R[0]
    y = R[1]
    f = v * (mass * ConvCoeffForce * v)

    dx = 4. * Vpt * x * (x**2 - x0**2)/x0**4 - f * alpha * y
    dy = f * y - f * alpha * x

    return np.array([dx, dy])

#%% 3. Reduced units of the committor_2.0 potentials
# committor_2.0 runs both potentials at kT = 1 in PLUMED natural units. Taking
# kT at 300 K as the energy unit, 1 angstroem and 1 amu, beta_eff = 1 reduced
# unit is reached at tau = hbar/kT = 25.5 fs.
DefaultTemperature = 300.0

def EnergyUnitAt(Temperature=DefaultTemperature):

    """

    :param1 Temperature: temperature in K.

    :return: E0: the energy that makes kT = 1 at that temperature, in cm-1.

    """

    return spc.physical_constants["Boltzmann constant in eV/K"][0]*Temperature/wvn2eV

#%% 4. Common interface
class Potential2D:

    """

    Analytic 2D potential in its own energy unit, Unit cm-1 each.

    A subclass implements Native and NativeGradient on arrays of shape
    (..., 2) and lists its basins, saddle guesses and grid. The reactant
    (q = 0) and product (q = 1) minima come first among the basins.

    """

    Name = ""
    UnitName = "cm-1"
    BasinGuesses = ()
    BasinNames = ()
    SaddleGuesses = ()
    GridLimits = ((-1.0, 1.0), (-1.0, 1.0))
    GridShape = (181, 181)
    BasinRadius = 0.2
    DefaultMass = 1.0
    # direction the mean path is binned along; None is the line from A to B
    PathAxis = None
    # harmonic walls of the PLUMED input, (kappa, lower, upper) on x
    Walls = None

    def __init__(self, Unit=1.0, Walls=True):

        """

        :param1 Unit:  size of one native energy unit, in cm-1.
        :param2 Walls: add the harmonic walls, if the potential has any (bool).

        """

        self.Unit = float(Unit)
        self.UseWalls = bool(Walls) and self.Walls is not None
        self._Minima = None
        self._Saddles = None

    # native surface, implemented by every subclass
    def Native(self, R):

        """

        :param1 R: coordinates, shape (..., 2), in angstroms.

        :return: V: potential in native units, shape (...).

        """

        raise NotImplementedError

    def NativeGradient(self, R):

        """

        :param1 R: coordinates, shape (..., 2), in angstroms.

        :return: G: gradient in native units per angstroem, shape (..., 2).

        """

        raise NotImplementedError

    # walls
    def WallEnergy(self, R):

        """

        Harmonic walls on x, zero inside, as LOWER_WALLS and UPPER_WALLS.

        :param1 R: coordinates, shape (..., 2).

        :return: V: wall energy in native units, shape (...).

        """

        x = np.asarray(R, dtype=float)[..., 0]
        if not self.UseWalls:
            return np.zeros_like(x)
        Kappa, Lower, Upper = self.Walls

        return Kappa*(np.minimum(x - Lower, 0.0)**2 + np.maximum(x - Upper, 0.0)**2)

    def WallGradient(self, R):

        """

        :param1 R: coordinates, shape (..., 2).

        :return: G: wall gradient in native units per angstroem, shape (..., 2).

        """

        R = np.asarray(R, dtype=float)
        G = np.zeros_like(R)
        if self.UseWalls:
            Kappa, Lower, Upper = self.Walls
            x = R[..., 0]
            G[..., 0] = 2.0*Kappa*(np.minimum(x - Lower, 0.0) + np.maximum(x - Upper, 0.0))

        return G

    # evaluation in every unit the callers need
    def Total(self, R):

        """

        :param1 R: coordinates, shape (..., 2).

        :return: V: potential plus walls in native units, shape (...).

        """

        return self.Native(R) + self.WallEnergy(R)

    def TotalGradient(self, R):

        """

        :param1 R: coordinates, shape (..., 2).

        :return: G: gradient of Total in native units per angstroem.

        """

        return self.NativeGradient(R) + self.WallGradient(R)

    def __call__(self, R):

        """

        :param1 R: 2D coordinates of one point, in angstroms.

        :return: E: potential energy in cm-1, as the samplers expect.

        """

        return float(self.Total(np.asarray(R, dtype=float)))*self.Unit

    def Energy(self, R):

        """

        :param1 R: coordinates, shape (..., 2).

        :return: E: potential energy in eV, shape (...).

        """

        return self.Total(R)*self.Unit*wvn2eV

    def Hessian(self, R, h=1.0e-5):

        """

        Hessian by central differences of the analytic gradient.

        :param1 R: 2D coordinates of one point.
        :param2 h: finite-difference step, in angstroms.

        :return: H: 2x2 Hessian in native units per angstroem squared.

        """

        R = np.asarray(R, dtype=float)
        H = np.zeros((2, 2))
        for i in range(2):
            e = np.zeros(2)
            e[i] = h
            H[:, i] = (self.TotalGradient(R + e) - self.TotalGradient(R - e))/(2.0*h)

        return 0.5*(H + H.T)

    # temperature
    def TauAtkT(self, kT):

        """

        :param1 kT: thermal energy in native units.

        :return: tau: time parameter at which the SWS ensemble has that kT, in fs.

        """

        return 1.0/(PECoeff*kT*self.Unit*wvn2eV)

    def kTAt(self, tau):

        """

        :param1 tau: time parameter tau (float), in fs.

        :return: kT: effective thermal energy in native units.

        """

        return 1.0/(PECoeff*tau*self.Unit*wvn2eV)

    @property
    def DefaultTau(self):

        """

        :return: tau: time parameter the committor is learned at, in fs.

        """

        raise NotImplementedError

    def DefaultScanTaus(self):

        """

        :return: Taus: time parameters of the default (tau, Ninter) scan, in fs.

        """

        raise NotImplementedError

    # stationary points
    def Minima(self):

        """

        Minima polished from the basin guesses, reactant first.

        :return: Points: array (NBasins, 2), in angstroms.

        """

        if self._Minima is None:
            self._Minima = np.array([spo.minimize(self.Total, g, jac=self.TotalGradient, method="BFGS",
                                                  options={"gtol": 1e-10}).x
                                     for g in self.BasinGuesses])

        return self._Minima

    def Saddles(self):

        """

        First-order saddles polished from the guesses by a root search on the gradient.

        :return: Points: array (NSaddles, 2), in angstroms.

        """

        if self._Saddles is None:
            Points = []
            for g in self.SaddleGuesses:
                Sol = spo.root(self.TotalGradient, g, jac=self.Hessian, tol=1e-12)
                w = np.linalg.eigvalsh(self.Hessian(Sol.x))
                if not Sol.success or not (w[0] < 0.0 < w[1]):
                    raise RuntimeError("%s: the guess %s did not converge to a first-order saddle"
                                       % (self.Name, str(g)))
                Points.append(Sol.x)
            self._Saddles = np.array(Points)

        return self._Saddles

    def StationaryPoints(self):

        """

        :return: Labels, Points: names and positions of the minima, then the saddles.

        """

        Labels = list(self.BasinNames) + ["S%i" % (k + 1) for k in range(len(self.Saddles()))]

        return Labels, np.concatenate([self.Minima(), self.Saddles()])

    def NormalModes(self, R, Mass):

        """

        Harmonic frequencies at a point, imaginary ones returned negative.

        :param1 R:    2D coordinates of the point.
        :param2 Mass: mass of the particle in a.m.u.s.

        :return: Nu, Periods: wavenumbers in cm-1 and vibrational periods in fs.

        """

        w = np.linalg.eigvalsh(self.Hessian(R)*self.Unit)
        # a force constant k in cm-1/A^2 has k = ConvCoeffForce m nu^2
        Nu = np.sign(w)*np.sqrt(np.abs(w)/(ConvCoeffForce*Mass))
        Periods = 1.0e15/(2.0*np.pi*spc.c*1.0e2*np.abs(Nu))

        return Nu, Periods

    def Describe(self, Mass=None):

        """

        Print the stationary points, their barriers and normal modes.

        :param1 Mass: mass entering the frequencies, or None for DefaultMass.

        :return: None.

        """

        Mass = self.DefaultMass if Mass is None else Mass
        Labels, Points = self.StationaryPoints()
        E0 = self.Total(Points[0])
        # h c nu/(2 pi k_B), the crossover temperature of a barrier in K cm
        TcCoeff = spc.h*spc.c*1.0e2/(2.0*np.pi*spc.k)
        print("[Potential] %s, 1 %s = %.3f cm-1 = %.5f eV; tau %.2f fs is kT_eff = %.4g %s"
              % (self.Name, self.UnitName, self.Unit, self.Unit*wvn2eV, self.DefaultTau,
                 self.kTAt(self.DefaultTau), self.UnitName))
        for Label, R in zip(Labels, Points):
            Nu, Periods = self.NormalModes(R, Mass)
            dE = self.Total(R) - E0
            Line = ("[Potential] %-2s (%+.4f, %+.4f)  V-V(A) = %10.4f %s = %6.2f kT_eff  nu = %+8.1f %+8.1f cm-1"
                    "  periods %6.2f %6.2f fs"
                    % (Label, R[0], R[1], dE, self.UnitName, dE/self.kTAt(self.DefaultTau),
                       Nu[0], Nu[1], Periods[0], Periods[1]))
            if Nu[0] < 0.0:
                Line += "  T_c = %.0f K" % (TcCoeff*abs(Nu[0]),)
            print(Line)

        return None

#%% 5. Proton transfer
class ProtonTransfer(Potential2D):

    """

    Symmetric double well along x coupled bilinearly to a harmonic bath along
    y, in cm-1. The only saddle sits at the origin.

    """

    Name = "2d"
    UnitName = "cm-1"
    BasinGuesses = ((-X0, 0.0), (X0, 0.0))
    BasinNames = ("A", "B")
    SaddleGuesses = ((0.0, 0.0),)
    GridLimits = ((-1.2, 1.2), (-0.8, 0.8))
    GridShape = (161, 121)
    BasinRadius = 0.22
    DefaultMass = mass
    # the proton-transfer coordinate
    PathAxis = (1.0, 0.0)

    def __init__(self, Vpt=Vpt, x0=X0, v=vbath, alpha=alpha, mass=mass, **Kwargs):

        """

        :param1 Vpt:   proton transfer barrier in cm-1.
        :param2 x0:    position of the minimum in angstroms.
        :param3 v:     harmonic oscillator frequency in cm-1.
        :param4 alpha: coupling between the bath and proton transfer.
        :param5 mass:  mass of the particle in a.m.u.s, which sets the bath force constant.

        """

        super().__init__(Unit=1.0, Walls=False)
        self.Parameters = dict(Vpt=Vpt, x0=x0, v=v, alpha=alpha, mass=mass)
        self.BasinGuesses = ((-x0, 0.0), (x0, 0.0))
        self.DefaultMass = mass

    def Native(self, R):

        R = np.asarray(R, dtype=float)

        return PotentialEnergy((R[..., 0], R[..., 1]), **self.Parameters)

    def NativeGradient(self, R):

        R = np.asarray(R, dtype=float)

        return np.moveaxis(PotentialGradient((R[..., 0], R[..., 1]), **self.Parameters), 0, -1)

    @property
    def DefaultTau(self):

        # puts the barrier near 8 kT_eff
        return 10.0

    def DefaultScanTaus(self):

        return list(np.linspace(0.1, 10., 10))

#%% 6. Reduced-unit potentials
class ReducedPotential(Potential2D):

    """

    A committor_2.0 potential, given in units of kT at a chosen temperature.

    """

    UnitName = "red"
    DefaultMass = 4.0
    ScankT = (10.0, 5.0, 3.0, 2.0, 1.5, 1.0)

    def __init__(self, Temperature=DefaultTemperature, Walls=True, **Kwargs):

        """

        :param1 Temperature: temperature in K whose kT is one reduced unit.
        :param2 Walls:       add the harmonic walls of the PLUMED input (bool).

        """

        super().__init__(Unit=EnergyUnitAt(Temperature), Walls=Walls)

    @property
    def DefaultTau(self):

        # the kT = 1 at which committor_2.0 runs
        return self.TauAtkT(1.0)

    def DefaultScanTaus(self):

        return [self.TauAtkT(kT) for kT in self.ScankT]

class MuellerBrown(ReducedPotential):

    """

    Mueller-Brown potential scaled by 0.15 and shifted by 146.7, so the
    reactant minimum sits at zero and the highest saddle near 16 kT.

    """

    Name = "muller"
    BasinGuesses = ((-0.558, 1.442), (0.623, 0.028), (-0.050, 0.467))
    BasinNames = ("A", "B", "I")
    SaddleGuesses = ((-0.822, 0.624), (0.212, 0.293))
    GridLimits = ((-1.5, 1.2), (-0.5, 2.2))
    BasinRadius = 0.15
    Walls = (1000.0, -1.3, 1.0)

    # amplitudes and quadratic forms of the four Gaussian terms
    Amp = np.array([-200.0, -100.0, -170.0, 15.0])
    a = np.array([-1.0, -1.0, -6.5, 0.7])
    b = np.array([0.0, 0.0, 11.0, 0.6])
    c = np.array([-10.0, -10.0, -6.5, 0.7])
    xc = np.array([1.0, 0.0, -0.5, -1.0])
    yc = np.array([0.0, 0.5, 1.5, 1.0])
    Scale = 0.15
    Shift = 146.7

    def _Terms(self, R):

        R = np.asarray(R, dtype=float)
        dx = R[..., 0, None] - self.xc
        dy = R[..., 1, None] - self.yc
        # the last term grows without bound, so it is capped far outside the walls
        T = self.Amp*np.exp(np.minimum(self.a*dx**2 + self.b*dx*dy + self.c*dy**2, 50.0))

        return T, dx, dy

    def Native(self, R):

        T, _, _ = self._Terms(R)

        return self.Scale*(self.Shift + T.sum(axis=-1))

    def NativeGradient(self, R):

        T, dx, dy = self._Terms(R)
        gx = (T*(2.0*self.a*dx + self.b*dy)).sum(axis=-1)
        gy = (T*(self.b*dx + 2.0*self.c*dy)).sum(axis=-1)

        return self.Scale*np.stack([gx, gy], axis=-1)

class TwoPaths(ReducedPotential):

    """

    Quartic double well with two channels between the basins, the upper one
    raised and the lower one lowered by Gaussians, as in committor_2.0.

    """

    Name = "two_paths"
    BasinGuesses = ((-1.033, -0.350), (1.122, 0.043))
    BasinNames = ("A", "B")
    # the upper channel, then the lower one, whose saddle is almost flat
    SaddleGuesses = ((-0.424, 0.637), (0.075, -0.995))
    GridLimits = ((-2.0, 2.0), (-2.0, 2.0))
    BasinRadius = 0.2
    Walls = (1000.0, -2.0, 2.0)

    # Gaussian amplitudes, centres and common width
    Amp = np.array([7.0, 1.0, -6.0])
    xg = np.array([-0.7, 1.0, -1.0])
    yg = np.array([0.8, -0.3, -0.6])
    Width = 0.4

    def _Gaussians(self, R):

        R = np.asarray(R, dtype=float)
        dx = R[..., 0, None] - self.xg
        dy = R[..., 1, None] - self.yg

        return self.Amp*np.exp(-(dx**2 + dy**2)/self.Width**2), dx, dy

    def Native(self, R):

        R = np.asarray(R, dtype=float)
        x = R[..., 0]
        y = R[..., 1]
        G, _, _ = self._Gaussians(R)

        return (10.0*(2.0 + 4.0/3.0*x**4 - 2.0*y**2 + y**4 + 10.0/3.0*x**2*(y**2 - 1.0))
                + G.sum(axis=-1))

    def NativeGradient(self, R):

        R = np.asarray(R, dtype=float)
        x = R[..., 0]
        y = R[..., 1]
        G, dx, dy = self._Gaussians(R)
        k = -2.0/self.Width**2
        gx = 10.0*(16.0/3.0*x**3 + 20.0/3.0*x*(y**2 - 1.0)) + (k*G*dx).sum(axis=-1)
        gy = 10.0*(-4.0*y + 4.0*y**3 + 20.0/3.0*x**2*y) + (k*G*dy).sum(axis=-1)

        return np.stack([gx, gy], axis=-1)

#%% 7. Registry
Potentials = {"2d": ProtonTransfer, "muller": MuellerBrown, "two_paths": TwoPaths}

def GetPotential(Name, Temperature=DefaultTemperature, Walls=True):

    """

    :param1 Name:        '2d', 'muller' or 'two_paths'.
    :param2 Temperature: temperature in K whose kT is one reduced unit; the
                         proton-transfer model is in cm-1 and ignores it.
    :param3 Walls:       add the harmonic walls of the PLUMED input (bool).

    :return: Potential: the Potential2D instance.

    """

    if Name not in Potentials:
        raise ValueError("unknown potential %s; choose one of %s" % (Name, ", ".join(Potentials)))

    return Potentials[Name](Temperature=Temperature, Walls=Walls)
