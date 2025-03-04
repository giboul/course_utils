"""
matplotlib==3.9.1
pandas==2.2.2
scipy==1.14.0
numpy==2.0.0

Script for estimating the h-Q relationship from a given profile (according to GMS). 
The object 'Profile' stores the hydraulic data as a pandas.DataFrame and creates a complete diagram with the .plot() method.

The following friction laws are supported here :
    - Gauckler-Manning-Strickler
    - Darcy
    - Chézy

Run script along with the following files to test:
    - profile.csv
    - closedProfile.csv
    - minimalProfile.csv

It will plot three diagrams with :
    - Limits enclosing the problem
    - The water_depth-discharge relation
    - The water_depth-critical_discharge relation
"""
from time import perf_counter
from typing import Iterable, Tuple
from pathlib import Path

from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from tkinter import filedialog, Tk, Frame, Entry, Label, Button, OptionMenu, StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np


g = 9.81


def GMS(K: float, Rh: float, i: float) -> float:
    """
    The Manning-Strickler equation

    Q = K * S * Rh^(2/3) * sqrt(i)

    Parameters
    ----------
    K : float
        The Manning-Strickler coefficient
    Rh : float
        The hydraulic radius, area/perimeter or width
    Js : float
        The slope of the riverbed

    Return
    ------
    float
        The discharge according to Gauckler-Manning-Strickler
    """
    return K * Rh**(2/3) * i**0.5


def equivalent_laws(Rh: float,
                    K: float = None,
                    C: float = None,
                    f: float = None) -> Tuple[float]:
    """
    Compute the equivalent coefficients for all three friction laws.
    Note that only one of the coefficients has to be specified, else 
    there is too much information.

    Parameters
    ----------
    Rh: float | np.ndarray
        The hydraulic radius.
    K: float = None
        The Gauckler-Manning-Strickler coefficient 1/n.
    C: float
        The Chézy coefficient.
    f: float
        The Darcy coefficient.

    Return
    ------
    K, C, f: Tuple[float | np.ndarray]
    """

    Rh = np.array(Rh)
    Rh[np.isclose(Rh, 0)] = None

    if sum(x is not None for x in (K, C, f)) != 1:
        raise ValueError("Specify exactly one of (K, C, f)")

    if K is not None:
        C = K * Rh**(1/6)
        f = 8*g / (K**2 * Rh**(1/3))
    elif C is not None:
        K = C / Rh**(1/6)
        f = 8 * g / C**2
    elif f is not None:
        K = (8*g/f)**0.5 / Rh**(1/6)
        C = (8*g/f)**0.5

    def array(a):
        if isinstance(a, (float, int)):
            a = np.full_like(Rh, a)
        a[np.isnan(a)] = 0
        return a

    K, C, f = map(array, (K, C, f))

    return K, C, f


def twin_points_old(x_arr: Iterable, z_arr: Iterable) -> Tuple[np.ndarray]:
    r"""
    Duplicate an elevation to every crossing of its level and the (x, z) curve.
    This will make for straight water tables when filtering like this :
    >>> z_masked = z[z <= z[ix]]  # array with z[ix] at its borders
    Thus, making the cross-section properties (S, P, B) easily computable.
    _                          ___
    /|     _____              ////
    /|    //////\            /////
    /+~~~+-------o~~~~~~~~~~+/////
    /|__//////////\        ///////
    ///////////////\______////////
    //////////////////////////////
    Legend:
         _
        //\ : ground
        ~ : water table
        o : a certain point given by some pair of (x, z)
        + : the new points created by this function

    Parameters
    ----------
    x : Iterable
        the horizontal coordinates array
    y : Iterable
        the vertical coordinates array

    Return
    ------
    np.ndarray
        the enhanced x-array
    np.ndarray
        the enhanced y-array
    """
    x_arr = np.array(x_arr, dtype=np.float32)
    z_arr = np.array(z_arr, dtype=np.float32)
    points = np.vstack((x_arr, z_arr)).T

    # to avoid looping over a dynamic array
    new_x = np.array([], dtype=np.float32)
    new_z = np.array([], dtype=np.float32)
    new_i = np.array([], dtype=np.int32)

    for i, line in enumerate(zip(points[:-1], points[1:]), start=1):

        (x1, z1), (x2, z2) = line

        if abs(z1-z2) < 1e-10:
            continue

        add_z = np.sort(z_arr[(min(z1, z2) < z_arr) & (z_arr < max(z1, z2))])
        if z2 < z1:  # if descending, reverse order
            add_z = add_z[::-1]
        add_x = x1 + (x2 - x1) * (add_z - z1)/(z2 - z1)
        add_i = np.full_like(add_z, i, dtype=np.int32)

        new_x = np.hstack((new_x, add_x))
        new_z = np.hstack((new_z, add_z))
        new_i = np.hstack((new_i, add_i))

    x = np.insert(x_arr, new_i, new_x)
    z = np.insert(z_arr, new_i, new_z)

    return x, z


def twin_points(x: Iterable, z: Iterable) -> Tuple[np.ndarray]:
    r"""
    Duplicate an elevation to every crossing of its level and the (x, z) curve.
    This will make for straight water tables when filtering like this :
    >>> z_masked = z[z <= z[ix]]  # array with z[ix] at its borders
    Thus, making the cross-section properties (S, P, B) easily computable.
    _                          ___
    /|     _____              ////
    /|    //////\            /////
    /+~~~+-------o~~~~~~~~~~+/////
    /|__//////////\        ///////
    ///////////////\______////////
    //////////////////////////////
    Legend:
         _
        //\ : ground
        ~ : water table
        o : a certain point given by some pair of (x, z)
        + : the new points created by this function

    Parameters
    ----------
    x : Iterable
        the horizontal coordinates array
    y : Iterable
        the vertical coordinates array

    Return
    ------
    np.ndarray
        the enhanced x-array
    np.ndarray
        the enhanced y-array
    """
    x = np.array(x, dtype=np.float32)
    z = np.array(z, dtype=np.float32)
    # Duplicate arrays to have matrices
    X = np.tile(x, (x.size, 1))
    Z = np.tile(z, (z.size, 1))
    # Find crossings
    C = (
        ((Z[:, :-1].T > z) & (Z[:, 1:].T < z)).T
        |
        ((Z[:, :-1].T < z) & (Z[:, 1:].T > z)).T
    )
    # Compute crossing coordinates
    Zc = (np.zeros_like(C).T + z).T
    Xc = np.full_like(X[:, :-1], float("nan"))
    Xc[C] = X[:, :-1][C] + (Zc[C] - Z[:, :-1][C]) * np.diff(X)[C]/np.diff(Z)[C]
    # Include original points (the last point is added afterwards)
    np.fill_diagonal(Xc, x)
    np.fill_diagonal(C, True)
    # Sort points to avoid half-turns
    axis = 0
    ix = Xc.argsort(axis=axis)
    ix[:, np.diff(x) < 0] = Xc[:, np.diff(x) < 0].argsort(axis=axis)[::-1]
    Xc = np.take_along_axis(Xc, ix, axis=axis)
    Zc = np.take_along_axis(Zc, ix, axis=axis)
    C = np.take_along_axis(C, ix, axis=axis)
    # Include last point and drop duplicates
    x = np.hstack((Xc.T[C.T], x[-1]))  # transpose for flattening order
    z = np.hstack((Zc.T[C.T], z[-1]))

    return x, z


def strip_outside_world(x: Iterable, z: Iterable) -> Tuple[np.ndarray]:
    r"""
    Return the same arrays without the excess borders
    (where the flow section width is unknown).

    If this is not done, the flow section could extend
    to the sides and mess up the polygon.

    This fuction assumes that twin_points has just been applied.

    Example of undefined profile:

             _
            //\~~~~~~~~~~~~~~~~~~  <- Who knows where this water table ends ?
           ////\          _
    ______//////\        //\_____   Legend:  _
    /////////////\______/////////           //\ : ground
    /////////////////////////////           ~ : water table

    Parameters
    ----------
    x : Iterable
        Position array from left to right
    z : Iterable
        Elevation array

    Return
    ------
    np.ndarray (1D)
        the stripped x
    np.ndarray(1D)
        the stripped y
    """
    x = np.array(x, dtype=np.float32)  # so that indexing works properly
    z = np.array(z, dtype=np.float32)
    ix = np.arange(x.size)  # indexes array
    argmin = z.argmin()  # index for the minimum elevation
    left = ix <= argmin  # boolean array inidcatinf left of the bottom
    right = argmin <= ix  # boolean array indicating right

    # Highest framed elevation (avoiding profiles with undefined borders)
    zmax = min(z[left].max(), z[right].max())
    assert zmax in z[left] and zmax in z[right]
    right_max_arg = argmin + (z[right] == zmax).argmax()
    left_max_arg = argmin - (z[left] == zmax)[::-1].argmax()
    right[right_max_arg+1:] = False
    left[:left_max_arg] = False

    return x[left | right], z[left | right]


def PSB_old(
    x_arr: Iterable,
    z_arr: Iterable,
    z: float
) -> Tuple[float]:  # Old version of PSB()
    """
    Return the polygon perimeter and area of the formed polygons.

    Parameters
    ----------
    x : Iterable
        x-coordinates
    y : Iterable
        y-coordinates
    z : float
        The z threshold (water table elevation)

    Return
    ------
    float
        Permimeter of the polygon
    float
        Surface area of the polygon
    float
        Length of the water table
    """
    x_arr = np.array(x_arr, dtype=np.float32)
    z_arr = np.array(z_arr, dtype=np.float32)

    mask = (z_arr[1:] <= z) & (z_arr[:-1] <= z)
    zm = (z_arr[:-1] + z_arr[1:])[mask]/2
    dz = np.diff(z_arr)[mask]
    dx = np.diff(x_arr)[mask]

    length = np.sqrt(dx**2 + dz**2).sum()
    surface = np.abs(((z - zm) * dx).sum())
    width = np.abs(dx.sum())

    return length, surface, width


def PSB(x: Iterable, z: Iterable):
    """
    Compute wet perimeter, wet surface and wet width for each of z's values.

    Parameters
    ----------
    x : Iterable
        x-coordinates
    z : Iterable
        y-coordinates

    Return
    ------
    NDarray
        Wet permimeter of the polygon
    NDarray
        Wet surface area of the polygon
    NDarray
        Wet width
    """
    x = np.array(x)
    z = np.array(z)
    X = np.tile(x, (x.size, 1))
    Z = np.tile(z, (z.size, 1))

    Zm = (Z[:, :-1] + Z[:, 1:])/2
    H = (z - Zm.T).T
    dZ = Z[:, 1:] - Z[:, :-1]
    dX = X[:, 1:] - X[:, :-1]
    dZ[H < 0] = 0.
    dX[H < 0] = 0.

    wet_perimeter = np.sqrt(dX**2 + dZ**2).sum(axis=1)
    wet_surface = (H*dX).sum(axis=1)
    wet_width = dX.sum(axis=1)

    return wet_perimeter, wet_surface, wet_width


def hydraulic_data(x: Iterable, z: Iterable) -> pd.DataFrame:
    """
    Derive relation between water depth and discharge (Manning-Strickler)

    Parameters
    ----------
    x : Iterable
        x (transversal) coordinates of the profile.
    z : Iterable
        z (elevation) coordinates of the profile.

    Return
    ------
    pandas.DataFrame
        x : x-coordinates
        z : z-coordinates
        P : wet perimeter
        S : wet surface
        B : dry perimeter
        h : water depth
        Qcr : critical discharge
    """
    # Compute wet section's properties
    P, S, B = PSB(x, z)
    # P, S, B = np.array([PSB_old(x, z, zi) for zi in z]).T

    h = z - z.min()

    Rh = np.full_like(P, None)
    Rh[np.isclose(S, 0)] = 0
    mask = ~ np.isclose(P, 0)
    Rh[mask] = S[mask] / P[mask]

    # Compute h_cr-Qcr
    Qcr = np.full_like(B, None)
    mask = ~ np.isclose(B, 0)
    Qcr[mask] = np.sqrt(g*S[mask]**3/B[mask])

    return pd.DataFrame.from_dict(dict(
        h=h, P=P, S=S, Rh=Rh, B=B, Qcr=Qcr
    ))


def profile_diagram(
    x: Iterable,
    z: Iterable,
    h: Iterable,
    Q: Iterable,
    Qcr: Iterable,
    fig=None,
    axes=None,
    *args,
    **kwargs
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot riverbed cross section and Q(h) in a sigle figure

    Parameters
    ----------
    h : float
        Water depth of stream cross section to fill
    show : bool
        wether to show figure or not
    fig, (axxz, axQh)
        figure and axes on which to draw (ax0: riverberd, ax1: Q(h))

    Returns
    -------
    pyplot figure
        figure containing plots
    pyplot axis
        profile coordinates transversal position vs. elevation
    pyplot axis
        discharge vs. water depth
    """
    if fig is None:
        fig = plt.figure(*args, **kwargs)
    if axes is None:
        axQh = fig.add_subplot()
        axxz = fig.add_subplot()
        axxz.patch.set_visible(False)
    else:
        axQh, axxz = axes

    x = np.array(x, dtype=np.float32)
    z = np.array(z, dtype=np.float32)
    h = np.array(h, dtype=np.float32)
    Q = np.array(Q, dtype=np.float32)
    Qcr = np.array(Qcr, dtype=np.float32)

    l1, = axxz.plot(x, z, '-ok',
                   mfc='w', lw=3, ms=5, mew=1,
                   label='Profil en travers utile')

    axxz.set_xlabel('Distance profil [m]')
    axxz.set_ylabel('Altitude [m.s.m.]')

    # positionning axis labels on right and top
    axxz.xaxis.tick_top()
    axxz.xaxis.set_label_position('top')
    axxz.yaxis.tick_right()
    axxz.yaxis.set_label_position('right')

    # plotting water depths
    ix = h.argsort()  # simply a sorting index
    l2, = axQh.plot(Q[ix], h[ix], '--b', label="$y_0$ (hauteur d'eau)")
    l3, = axQh.plot(Qcr[ix], h[ix], '-.', label='$y_{cr}$ (hauteur critique)')
    axQh.set_xlabel('Débit [m$^3$/s]')
    axQh.set_ylabel("Hauteur d'eau [m]")
    axxz.grid(False)

    # plotting 'RG' & 'RD'
    ztxt = (z.max() + z.min())/2
    axxz.text(x.min(), ztxt, 'RG')
    axxz.text(x.max(), ztxt, 'RD', ha='right')

    # match height and altitude ylims
    axQh.set_ylim(axxz.get_ylim() - z.min())

    # common legend for both axes
    lines = (l1, l2, l3)
    labels = [line.get_label() for line in lines]
    axxz.legend(lines, labels)

    axQh.dataLim.x0 = 0
    axQh.autoscale_view()

    return fig, (axxz, axQh)


class Profile(pd.DataFrame):
    """
    An :func:`~pandas.DataFrame` object.

    Attributes
    ----------
    x : pd.Series
        x-coordinates 
        (horizontal distance from origin)
    z : pd.Series
        z-coordinates (altitudes)
    h : pd.Series
        Water depths
    P : pd.Series
        Wtted perimeter
    S : pd.Series
        Wetted area
    Rh : pd.Series
        Hydraulic radius
    Q : pd.Series
        Discharge (GMS)
    Q : pd.Series
        Critical discharge
    K : float
        Manning-Strickler coefficient
    Js : float
        bed's slope

    Methods
    -------
    plot(h: float = None)
        Plots a matplotlib diagram with the profile,
        the Q-h & Q-h_critical curves and a bonus surface from h
    interp_h_vs_Q(h: Iterable)
        Returns an quadratic interpolation of the discharge (GMS)
    """

    def __init__(
        self,
        x: Iterable,  # position array from left to right river bank
        z: Iterable,  # altitude array from left to right river bank
        **fric_kwargs
    ) -> None:
        """
        Initialize :func:`~hydraulic_data(x, z, K, Js)` and set the friction law Js

        Parameters
        ----------
        x: Iterable
            position array from left to right river bank
        z: Iterable
            altitude array from left to right river bank
        K: float = None
            The manning-strickler coefficient
        C: float = None
            The Chézy coefficient
        f: float = None
            The Darcy-Weisbach coefficient
        Js: float = None
            The riverbed's slope
        """

        x, z = twin_points(x, z)
        x, z = strip_outside_world(x, z)
        df = pd.DataFrame.from_dict(dict(x=x, z=z))
        hd = hydraulic_data(x, z)
        df = pd.concat((df, hd), axis="columns")

        super().__init__(df)

        if fric_kwargs:
            Js = fric_kwargs.pop("Js")
            if isinstance(Js, float):
                Js = np.full(self.x.size, Js)
            K, C, f = equivalent_laws(self.Rh, **fric_kwargs)
            self["v"] = GMS(K, self.Rh, Js)
            self["Q"] = self.S * self.v

            self["K"] = K
            self["Js"] = Js

        self = self.sort_values("h").reset_index()

    def interp_h_vs_K(self, h_array: Iterable) -> np.ndarray:
        return interp1d(self.h, self.K, assume_sorted=False)(h_array)

    def interp_h_vs_Js(self, h_array: Iterable) -> np.ndarray:
        return interp1d(self.h, self.Js, assume_sorted=False)(h_array)

    def interp_h_vs_B(self, h_array: Iterable) -> np.ndarray:
        return interp1d(self.h, self.B, assume_sorted=False)(h_array)

    def interp_h_vs_P(self, h_array: Iterable) -> np.ndarray:
        return interp1d(self.h, self.P, assume_sorted=False)(h_array)

    def interp_h_vs_S(self, h_array: Iterable) -> np.ndarray:
        """
        Quadratic interpolation of the surface. 
        dS = dh*dB/2 where B is the surface width

        Parameters
        ----------
        h_array : Iterable
            Array of water depths

        Returns
        -------
        np.ndarray
            The corresponding surface area
        """

        h, B, S = self[
            ["h", "B", "S"]
        ].sort_values("h").drop_duplicates("h").to_numpy().T

        s = np.zeros_like(h_array)
        for i, h_interp in enumerate(h_array):
            # Checking if h_interp is within range
            mask = h >= h_interp
            if mask.all():
                s[i] = 0
                continue
            if not mask.any():
                s[i] = float("nan")
                continue

            # Find lower and upper bounds
            argsup = mask.argmax()
            arginf = argsup - 1
            # Interpolate
            r = (h_interp - h[arginf]) / (h[argsup] - h[arginf])
            Bi = r * (B[argsup] - B[arginf]) + B[arginf]
            ds = (h_interp - h[arginf]) * (Bi + B[arginf])/2
            s[i] = S[arginf] + ds

        return s

    def interp_h_vs_Q(self, h_array: Iterable) -> np.ndarray:
        """
        Interpolate discharge from water depth with
        the quadratic interpolation of S.

        Parameters
        ----------
        h_array : Iterable
            The water depths array.

        Return
        ------
        np.ndarray
            The corresponding discharges
        """
        h = np.array(h_array, dtype=np.float32)
        S = self.interp_h_vs_S(h)
        P = self.interp_h_vs_P(h)
        Q = np.zeros_like(h)
        mask = ~np.isclose(P, 0)
        Q[mask] = S[mask] * GMS(
            self.interp_h_vs_K(h)[mask],
            S[mask]/P[mask],
            self.interp_h_vs_Js(h)[mask]
        )
        return Q

    def interp_h_vs_Qcr(self, h_array: Iterable) -> np.ndarray:
        """
        Interpolate critical discharge from water depth.

        Parameters
        ----------
        h_array : Iterable
            The water depths array.

        Return
        ------
        np.ndarray
            The corresponding critical discharge
        """
        Qcr = np.full_like(h_array, None)
        B = self.interp_h_vs_B(h_array)
        S = self.interp_h_vs_S(h_array)
        mask = ~ np.isclose(B, 0)
        Qcr[mask] = np.sqrt(g*S[mask]**3/B[mask])
        return Qcr

    def plot(self, interp_num=1000, *args, **kwargs) -> Tuple[Figure, Tuple[plt.Axes]]:
        """Call :func:`~profile_diagram(self.x, self.z,  self.h, self.Q, self.Qcr)` 
        and update the lines with the interpolated data."""
        fig, (ax1, ax2) = profile_diagram(
            self.x, self.z, self.h, self.Q, self.Qcr,
            *args, **kwargs
        )

        l1, l2 = ax2.get_lines()
        h = np.linspace(self.h.min(), self.h.max(), interp_num)
        l1.set_data(self.interp_h_vs_Q(h), h)
        l2.set_data(self.interp_h_vs_Qcr(h), h)

        return fig, (ax1, ax2)


def csv_to_csv(input_file: str,
               xcol="x",
               zcol="z",
               output_file: str = None,
               Js: float = 0.12/100,
               K: float = None,
               C: float = None,
               f: float = None) -> None:
    """
    From a table with the profile, write a table with hydraulic details.
    
    Paramters
    ---------
    input_file: str | pathlib.Path
        The input file with columns `xcol` and `zcol`
    xcol: str
        The name of the column with the x-coordinates
    zcol: str
        The name of the column with the z-coordinates
    output_file: str = None
        Name of the output file with hydraulic details.
    Js: float
        The slope of the bed
    K: float = None
        The Gauckler-Manning-Strickler coefficient
    C: float = None
        The Chézy coefficient
    f: float = None
        The Darcy coefficient
    """
    if sum(v is None for v in (K, C, f)) != 1:
        raise ValueError("Speficify exactly one of 'K', 'C' or 'f'.")

    input_file = Path(input_file)
    if output_file is None:
        output_file = input_file.parent / input_file.stem
        output_file = output_file / "-processed.csv"
    else:
        output_file = Path(output_file)

    df = pd.read_csv(input_file)
    profile = Profile(df[xcol], df[zcol], Js=Js, K=K, C=C, f=f)
    profile.to_csv(output_file, index=False)


def select_file():
    """Select the path of the disired file though a file dialog."""
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(parent=root, title='Choose file')


def app():
    """
    Making an app from all above functions.

    +-------------------+-------------------------+-------------+
    |       Info        | pd.read_csv(path='...') | file dialog |
    +-------------------+-------------------------+-------------+
    |        x          |          Entry          |             |
    +-------------------+-------------------------+             |
    |        z          |          Entry          |             |
    +-------------------+-------------------------+             |
    | Friction law menu |          Entry          |    PLOT     |
    +-------------------+-------------------------+             |
    |       Slope       |          Entry          |             |
    +-------------------+-------------------------+             |
    |    Save Button    |          Path           |             |
    +-------------------+-------------------------+-------------+
    """

    params = dict(
        ipath = Path(__file__).parent / "profiles" / "profile.csv",
        opath = Path(__file__).parent / "profiles" / "profile-processed.csv",
        xcol="x-coordinate column",
        zcol="z-coordinates column",
        friction="33",
        friclaw="GMS",
        slope=str(1.2/1000),
        pandas_kwargs = dict()
    )

    def log(s):
        lab1text.set(s)
        lab1.update_idletasks()

    # Setting up window and frame
    window = Tk()
    window.title("Courbe de tarage")
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    width = int(screen_width*0.5)
    height = int(screen_height*0.5)
    left = (screen_width - width) // 2
    top = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{left}+{top}")

    # Initialize main frame
    frame = Frame(window, borderwidth=10)
    frame.pack(fill='both', expand=True)
    for col, w in enumerate((5, 5, 50, 1, 5)):
        frame.columnconfigure(col, pad=10, weight=w)
    for row, w in enumerate((1, 3, 3, 3, 3, 3, 2)):
        frame.rowconfigure(row, pad=5, weight=w)

    # Input file path and arguments to pandas
    def validate_kwargs(P):
        if P == '':
            return True

        try:
            kwargs = eval(f"dict({P})")
        except:
            log()
            filetxt.config(bg="red")
            return True

        kwargs["path"] = Path(kwargs["path"])
        test = kwargs["path"].is_file()
        if test is False:
            filetxt.config(bg="red")
        else:
            filetxt.config(bg="white")
            params["ipath"] = kwargs.pop("path")
            params["pandas_kwargs"] = kwargs

        return True

    # Simple label
    lab1basetext = "Click on figure or\n hit enter to reload it"
    lab1text = StringVar(frame, lab1basetext)
    lab1 = Label(frame, textvariable=lab1text)
    lab1.grid(row=0, column=0, sticky="NSWE")

    # Widgets for file loading
    pd_lab = Label(frame, text="pd.read_csv(")
    pd_endlab = Label(frame, text=")")

    filetxt = Entry(frame)
    filetxt.insert(0, f"path='{params['ipath']}'")
    filetxt.config(validate="key", vcmd=(frame.register(validate_kwargs), "%P"))

    # For changing files within the app
    def update_path():
        path = select_file()
        filetxt.delete(0, 'end')
        filetxt.insert(0, f"path='{path}'")
        params["ipath"] = path
        replot()
    navigatebutt = Button(frame, text="Navigate files", command=lambda: update_path())

    # Specifying the columns to use for coordinates reading
    def update_dict(key: str):
        def uptd(val):
            params[key] = val
            return True
        return (frame.register(uptd), "%P")
    xcollab = Label(frame, text=params["xcol"])
    xcoltxt = Entry(frame, validate="key", vcm=update_dict("xcol"))
    xcoltxt.insert(0, 'Dist. cumulée [m]')

    zcollab = Label(frame, text=params["zcol"])
    zcoltxt = Entry(frame, validate="key", vcm=update_dict("zcol"))
    zcoltxt.insert(0, 'Altitude [m.s.m.]')

    # Friction parameters widgets
    fricvar = StringVar(frame)
    fricvar.set(params["friclaw"])
    def update_friclaw(val):
        params["friclaw"] = val
    friclab = OptionMenu(frame, fricvar, "GMS", "Chézy", "Darcy", command=update_friclaw)
    fric_coefs = {"GMS": "K", "Chézy": "C", "Darcy": "f"}

    def uptd_fric(val):
        try:
            val = float(val)
            params["friction"] = val
            frictxt.config(bg="white")
        except ValueError:
            frictxt.config(bg="red")
        return True
    frictxt = Entry(frame, validate="key", vcm=(frame.register(uptd_fric), "%P"), justify="right")
    frictxt.insert(0, params["friction"])

    # Bed slope entry widget
    def uptd_slope(val):
        try:
            val = float(val)
            params["slope"] = val
            slopetxt.config(bg="white")
        except ValueError:
            slopetxt.config(bg="red")
        return True
    slopelab = Label(frame, text="Slope")
    slopetxt = Entry(frame, validate="key", vcm=(frame.register(uptd_slope), "%P"), justify="right")
    slopetxt.insert(0, params["slope"])

    # For saving the results in a csv table
    def validate_opath(P):
        P = Path(P)
        if P.parent.is_dir():
            params["opath"] = P
            savetxt.config(bg="white")
        else:
            savetxt.config(bg="red")
        return True
    def save_df():
        df = pd.read_csv(params["ipath"])
        profile = Profile(df[params["xcol"]],
                          df[params["zcol"]],
                          Js=float(params["slope"]),
                          **{fric_coefs[params["friclaw"]]: float(params["friction"])})
        profile.to_csv(params["opath"])
    savebutt = Button(frame, text="Save results", command=save_df)
    savetxt = Entry(frame, validate="key", vcmd=(frame.register(validate_opath), "%P"))
    savetxt.insert(0, params["opath"])

    # Widget Placement
    pd_lab.grid(row=0, column=1, sticky="NSE")
    pd_endlab.grid(row=0, column=3, sticky="NSW")
    filetxt.grid(row=0, column=2, sticky="NSWE")
    navigatebutt.grid(row=0, column=4)

    xcollab.grid(row=1, column=0, sticky="NSWE")
    xcoltxt.grid(row=1, column=1, sticky="NSWE")

    zcollab.grid(row=2, column=0, sticky="NSWE")
    zcoltxt.grid(row=2, column=1, sticky="NSWE")

    friclab.grid(row=3, column=0, sticky="NSWE")
    frictxt.grid(row=3, column=1, sticky="NSWE")

    slopelab.grid(row=4, column=0, sticky="NSWE")
    slopetxt.grid(row=4, column=1, sticky="NSWE")

    savebutt.grid(row=5, column=0, sticky="NSWE")
    savetxt.grid(row=5, column=1, sticky="NSWE")

    # Figure setup
    with plt.style.context('ggplot'):
        fig = plt.figure(layout="tight")
        ax1 = fig.add_subplot()
        ax2 = fig.add_subplot()

    def replot(event=None):

        with plt.style.context('ggplot'):
            log("Clearing axes...")
            ax1.cla()
            ax2.cla()
            ax2.patch.set_visible(False)

            log("Reading csv file...")
            try:
                df = pd.read_csv(params["ipath"], **params["pandas_kwargs"])
            except:
                log("Error while reading the file.\nSee in the terminal.")

            log("Checking column names...")
            nocols = False
            if params["xcol"] not in df.columns:
                xcoltxt.config(bg="red")
                nocols = True
            else:
                xcoltxt.config(bg="white")
            if params["zcol"] not in df.columns:
                zcoltxt.config(bg="red")
                nocols = True
            else:
                zcoltxt.config(bg="white")
            if nocols:
                log("Did not find the given columns names in file.")
                return False

            log("Computing hydraulic properties...")
            profile = Profile(df[params["xcol"]],
                        df[params["zcol"]],
                        Js=float(params["slope"]),
                        **{fric_coefs[params["friclaw"]]: float(params["friction"])})
            log("Plotting figure...")
            profile.plot(fig=fig, axes=(ax1, ax2))
            canvas.draw()
            log(lab1basetext)

    canvas = FigureCanvasTkAgg(fig, frame)
    toolbar_frame = Frame(frame)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar_frame.grid(row=6, column=0, columnspan=2, sticky="NW")
    canvas.get_tk_widget().grid(row=1, column=2, rowspan=6, columnspan=3, sticky="NSWE")
    canvas.mpl_connect("button_press_event", replot)
    window.bind_all('<Return>', replot)

    window.mainloop()


def main():
    df = pd.read_csv(Path(__file__).parent / "profiles" / "profile.csv")
    fig, (ax1, ax2) = Profile(df["Dist. cumulée [m]"], df["Altitude [m.s.m.]"], K=33, Js=0.0012).plot()
    plt.show()

    df = pd.read_csv(Path(__file__).parent / "profiles" / "closedProfile.csv")
    prof = Profile(df.x, df.z, K=33, Js=0.0012)
    fig, (ax1, ax2) = prof.plot()
    ax2.set_xlim((prof.Q.min(), prof.Q.max()))
    plt.show()

    df = pd.read_csv(Path(__file__).parent / "profiles" / "minimalProfile.csv")
    fig, (ax1, ax2) = Profile(df.x, df.z, K=33, Js=0.0012).plot()
    plt.show()

    df = pd.read_csv(Path(__file__).parent / "profiles" / "randomProfile.csv")
    fig, (ax1, ax2) = Profile(df.x, df.z, K=33, Js=0.0012).plot()
    plt.show()


def benchmark():
    fnames = ["profile.csv", "closedProfile.csv", "minimalProfile.csv", "randomProfile.csv"]
    dfs = [pd.read_csv(Path(__file__).parent / f) for f in fnames]
    n = 1000
    for i, df in enumerate(dfs):
        xarr, zarr = df.to_numpy().T[:2]
        print(f"### {i} ###")
        a = perf_counter()
        for _ in range(n):
            x, z = twin_points(xarr, zarr)
        b = perf_counter()
        for _ in range(n):
            PSB(x, z)
        c = perf_counter()
        for _ in range(n):
            x, z = twin_points_old(xarr, zarr)
        d = perf_counter()
        for _ in range(n):
            [PSB_old(x, z, zi) for zi in z]
        e = perf_counter()

        print(f"twin: {b-a:.2f} s")
        print(f"otwi: {d-c:.2f} s")
        print("---")
        print(f"PSB:  {c-b:.2f} s")
        print(f"oPS:  {e-d:.2f} s")


if __name__ == "__main__":
    app()
    # benchmark()
