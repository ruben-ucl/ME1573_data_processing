import numpy as np
import matplotlib.pyplot as plt

# Set up figure
fig = plt.figure(figsize=(10, 15), dpi=300, tight_layout=True)
ax = fig.add_subplot(projection='3d')

# Define melt pool dimensions
L = 250
d = 50
w = 70

# Draw front ellipsoid segment
def generate_ellipsoid(a, b, c, u, v):
    uu = np.linspace(u[0], u[1], 100)
    vv = np.linspace(v[0], v[1], 100)
    x = a * np.outer(np.cos(uu), np.sin(vv))
    y = b * np.outer(np.sin(uu), np.sin(vv))
    z = c * np.outer(np.ones_like(uu), np.cos(vv))
    return (x, y, z)
    
ellipse = generate_ellipsoid(d, w/2, d, (-np.pi/2, np.pi/2), (np.pi/2, np.pi))
ax.plot_surface(ellipse[0], ellipse[1], ellipse[2], alpha=0.5, color='r')

# Draw rear paraboloid segment
def generate_paraboloid(a, b, c):
    c1 = np.sqrt(-a/b**2)
    c2 = np.sqrt(-a/c**2)
    c3 = a
    y = np.linspace(-b, b, 500)
    z = np.linspace(c, 0, 500)
    y, z = np.meshgrid(y, z)
    x = (c1*y)**2 + (c2*z)**2 + c3
    for i in np.arange(len(y)):
        for j in np.arange(len(z)):
            if x[i, j] >= 0 and x[i, j] < 10:
                x[i, j] = 0
            elif x[i, j] >= 10:
                x[i, j] = np.nan
            else:
                pass
    return (x, y, z)
    
paraboloid = generate_paraboloid(d-L, w/2, -d)
ax.plot_surface(paraboloid[0], paraboloid[1], paraboloid[2], alpha=0.5, color='r')

# Draw outlines
def generate_yzellipse(a, b):
    y = np.linspace(-a/2, a/2, 100)
    z = -(b/(a/2)) * np.sqrt((a/2)**2 - y**2)
    x = [0 for i in z]
    return (x, y, z)

ellipse = generate_yzellipse(w, d)
ax.plot(ellipse[0], ellipse[1], ellipse[2], color='k', linewidth=0.5, linestyle='--')

def generate_xyellipse(a, b):
    y = np.linspace(-a/2, a/2, 100)
    x = (b/(a/2)) * np.sqrt((a/2)**2 - y**2)
    z = [0 for i in x]
    return (x, y, z)

ellipse = generate_xyellipse(w, d)
ax.plot(ellipse[0], ellipse[1], ellipse[2], color='k', linewidth=0.5, linestyle='--')

def generate_xzellipse(a, b):
    x = np.linspace(0, a, 100)
    z = -(b/(a)) * np.sqrt((a)**2 - x**2)
    y = [0 for i in x]
    return (x, y, z)

ellipse = generate_xzellipse(d, d)
ax.plot(ellipse[0], ellipse[1], ellipse[2], color='k', linewidth=0.5, linestyle='--')

def generate_xzparabola(a, b):
    z = np.linspace(-a, 0, 100)
    x = -b/a**2 * z**2 + b
    y = [0 for i in x]
    return (x, y, z)
    
parabola = generate_xzparabola(d, d-L)
ax.plot(parabola[0], parabola[1], parabola[2], color='k', linewidth=0.5, linestyle='--')

def generate_xyparabola(a, b):
    y = np.linspace(-a/2, a/2, 100)
    x = -b/(a/2)**2 * y**2 + b
    z = [0 for i in x]
    return (x, y, z)
    
parabola = generate_xyparabola(w, d-L)
ax.plot(parabola[0], parabola[1], parabola[2], color='k', linewidth=0.5, linestyle='--')

# Draw orthogonal guides    
def generate_x_guide(start, stop):
    x = np.arange(start, stop)
    y = z = np.array([0 for e in x])
    return (x, y, z)
    
def generate_y_guide(start, stop):
    y = np.arange(start, stop)
    x = z = np.array([0 for e in y])
    return (x, y, z)
    
def generate_z_guide(start, stop):
    z = np.arange(start, stop)
    x = y = np.array([0 for e in z])
    return (x, y, z)

x_guide = generate_x_guide(d-L, d)       
y_guide = generate_y_guide(-w/2, w/2)
z_guide = generate_z_guide(-d, 0)

ax.plot(x_guide[0], x_guide[1], x_guide[2], linewidth=0.6)
ax.plot(y_guide[0], y_guide[1], y_guide[2], linewidth=0.6)
ax.plot(z_guide[0], z_guide[1], z_guide[2], linewidth=0.6)

ax_range = L + 40
ax.set_xlim(d-L-20, w/2+20)
ax.set_ylim(-ax_range/2, ax_range/2)
ax.set_zlim(-ax_range/2, ax_range/2)
ax.set_axis_off()

plt.show()