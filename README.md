# Solving PDEs using Physics-Informed Neural Networks (PINNs)

This repository contains examples of solving various partial differential equations (PDEs) using **Physics-Informed Neural Networks (PINNs)**.

## 📌 PDE Examples

### 1️⃣ Burgers' Equation
- **Description:** A fundamental nonlinear PDE that models fluid dynamics and shock waves.
- **Results:**
  
 <table>
  <tr>
    <td><img src="burgers_equation /plot/pred.png" width="100%"/><br><center>Figure 1: Burgers' Equation Results.</center></td>
  </tr>
</table>

### 2️⃣ Navier-Stokes Equation
- **Description:** Governs the motion of fluid substances and is widely used in fluid mechanics.
- **Results:**
   <table>
    <tr>
      <td><img src="naiver_stoke/cylinder wake gif.gif" width="100%"/><br><center>Figure 1: Navier-Stokes Equation Results.</center></td>
    </tr>
   </table>
 

### 3️⃣ Heat Equation
- **Description:** Describes the distribution of heat in a given region over time.
- **Results:**
   <table>
  <tr>
    <td><img src="heat_equation/Copy of passive_tracer_spectral_pinns (1).gif" width="100%"/><br><center>Figure 1: Heat Equation Results.</center></td>
  </tr>
</table>

## 🛠️ Dependencies
To run these examples, install the required dependencies:
```bash
pip install deepxde torch numpy matplotlib
```

