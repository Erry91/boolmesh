//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use std::f64::consts::PI;
use std::time::Instant;
use bevy::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::light::CascadeShadowConfigBuilder;
use bevy::render::render_resource::{Extent3d, PrimitiveTopology, TextureDimension, TextureFormat};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use boolmesh::prelude::*;
use boolmesh::Real;

type Uv = boolmesh::Vec2;
type Buf = (Vec<f64>, Vec<usize>, Vec<Uv>);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut cmds: Commands,
    mut mats: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
) {
    let now = Instant::now();

    let num = 4;
    let res = menger_sponge(num);

    println!(">>>>>>>>>>>>>> Compute a menger sponge of level {}, elapsed time: {:?}", num, now.elapsed());

    cmds.spawn((
        DirectionalLight { shadows_enabled: true, ..default() },
        Transform::from_xyz(2., 3., 1.5).looking_at(Vec3::ZERO, Vec3::Y),
        CascadeShadowConfigBuilder { maximum_distance: 6., first_cascade_far_bound: 3., ..default() }.build(),
    ));
    let focus = Vec3::new(0., -0.14, 0.);
    cmds.spawn((Transform::from_translation(focus + Vec3::new(1.6, 1.1, 1.6)).looking_at(focus, Vec3::Y), PanOrbitCamera { focus, ..default() }));

    let (mut pos, mut vns, mut uvs) = (vec![], vec![], vec![]);
    for ((hs, vs), n) in res.hs.chunks(3).zip(res.vals.chunks(3)).zip(&res.fns) {
        for k in 0..3 {
            let p = res.ps[hs[k].tail];
            pos.push([p.x as f32, p.y as f32, p.z as f32]);
            vns.push([n.x as f32, n.y as f32, n.z as f32]);
            uvs.push([vs[k].x as f32, vs[k].y as f32]);
        }
    }

    let m = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vns)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs);

    cmds.spawn((
        Mesh3d(meshes.add(m)),
        MeshMaterial3d(mats.add(StandardMaterial {
            base_color_texture: Some(images.add(texture())),
            perceptual_roughness: 0.8,
            ..default()
        })),
    ));
}

pub fn menger_sponge(n: usize) -> Manifold<Uv> {
    let (pos, idx, uvs) = cube([0.; 3], [1.; 3], skin_uv);
    let res = Manifold::new_with_values(&pos, &idx, &uvs).unwrap();
    let mut holes = (vec![], vec![], vec![]);
    fractal(&mut holes, 0., 0., 1., 1, n);
    let holes_z = Manifold::new_with_values(&holes.0, &holes.1, &holes.2).unwrap();

    let rot = |rx: f64, ry: f64, rz: f64| {
        let r = boolmesh::Mat3::from_euler(glam::EulerRot::XYZ, rx as Real, ry as Real, rz as Real);
        let snap = |v: Real| if (v - 0.5).abs() < 1e-4 { 0.5 } else if (v + 0.5).abs() < 1e-4 { -0.5 } else { v as f64 };
        let mut flat = vec![];
        for h in holes_z.hs.iter() {
            let p = r * holes_z.ps[h.tail];
            flat.extend_from_slice(&[snap(p.x), snap(p.y), snap(p.z)]);
        }
        let idx = (0..holes_z.hs.len()).collect::<Vec<_>>();
        Manifold::new_with_values(&flat, &idx, &holes_z.vals).unwrap()
    };

    let holes_x = rot(PI / 2., 0., 0.);
    let holes_y = rot(0., PI / 2., 0.);

    let res = compute_boolean(&res, &holes_z, OpType::Subtract).unwrap();
    let res = compute_boolean(&res, &holes_x, OpType::Subtract).unwrap();
    let res = compute_boolean(&res, &holes_y, OpType::Subtract).unwrap();
    res
}

// The texture is an atlas of two squares: the outer skin on the left and the tunnel wall on the right.
const PAD: f64 = 0.004;
fn atlas(i: usize, s: f64) -> Real { ((i as f64 + PAD + (1. - 2. * PAD) * s) / 2.) as Real }

fn skin_uv(k: usize, q: [f64; 3]) -> Uv { Uv::new(atlas(0, q[(k + 1) % 3]), q[(k + 2) % 3] as Real) }

fn tunnel_uv(k: usize, q: [f64; 3]) -> Uv { Uv::new(atlas(1, if k == 0 { q[1] } else { q[0] }), q[2] as Real) }

fn cube(c: [f64; 3], s: [f64; 3], uv: fn(usize, [f64; 3]) -> Uv) -> Buf {
    let (mut pos, mut idx, mut uvs) = (vec![], vec![], vec![]);
    for k in 0..3 { for d in [1., 0.] {
        let base = pos.len() / 3;
        for (a, b) in [(0., 0.), (1., 0.), (1., 1.), (0., 1.)] {
            let mut q = [0.; 3];
            q[k] = d;
            q[(k + 1) % 3] = a;
            q[(k + 2) % 3] = b;
            pos.extend((0..3).map(|i| c[i] + s[i] * (q[i] - 0.5)));
            uvs.push(uv(k, q));
        }
        idx.extend(if d > 0. { [0, 1, 2, 0, 2, 3] } else { [0, 2, 1, 0, 3, 2] }.map(|i| base + i));
    }}
    (pos, idx, uvs)
}

pub fn fractal(holes: &mut Buf, x: f64, y: f64, w: f64, depth: usize, depth_max: usize) {
    let w = w / 3.;
    let (pos, idx, uvs) = cube([x, y, 0.], [w, w, 1.], tunnel_uv);
    let offset = holes.0.len() / 3;
    holes.0.extend(pos);
    holes.1.extend(idx.iter().map(|i| i + offset));
    holes.2.extend(uvs);

    if depth == depth_max { return; }

    for xy in [
        (x - w, y - w),
        (x - w, y    ),
        (x - w, y + w),
        (x    , y + w),
        (x + w, y + w),
        (x + w, y    ),
        (x + w, y - w),
        (x    , y - w)
    ] {
        fractal(holes, xy.0, xy.1, w, depth + 1, depth_max);
    }
}

fn texture() -> Image {
    let h = 1024;
    let mut data = Vec::with_capacity(h * h * 8);
    for y in 0..h { for x in 0..2 * h {
        let s = (((x % h) as f64 + 0.5) / h as f64 - PAD) / (1. - 2. * PAD);
        let (s, t) = (s as f32, (y as f32 + 0.5) / h as f32);
        let c = if x < h { skin(s, t) } else { tunnel(s, t) }.to_srgba();
        data.extend([c.red, c.green, c.blue].map(|v| (v.clamp(0., 1.) * 255.).round() as u8));
        data.push(255);
    }}
    Image::new(
        Extent3d { width: 2 * h as u32, height: h as u32, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}

fn skin(s: f32, t: f32) -> Color {
    let n = 27.;
    let (i, j) = ((s * n).floor(), (t * n).floor());
    let (fs, ft) = (s * n - i, t * n - j);
    if fs.min(1. - fs).min(ft).min(1. - ft) < 0.05 { return Color::srgb(0.06, 0.05, 0.05); }
    let r = (((i + 0.5) / n - 0.5).powi(2) + ((j + 0.5) / n - 0.5).powi(2)).sqrt();
    let jitter = ((i * 12.9898 + j * 78.233).sin() * 43758.545).fract() * 0.08;
    let hue = 190. - 300. * r;
    let light = 0.45 + 0.12 * (r * 40.).cos() + jitter;
    Color::hsl(hue.rem_euclid(360.), 0.65, light)
}

fn tunnel(s: f32, t: f32) -> Color {
    let band = (s * 3.).floor();
    let f = s * 3. - band;
    let ring = (t * 27.).fract();
    let edge = f.min(1. - f) < 0.04 || ring.min(1. - ring) < 0.03;
    let ao = 0.2 + 0.8 * (2. * t - 1.).abs().powf(0.6);
    let base = if band == 1. { Color::srgb(0.85, 0.42, 0.22) } else { Color::srgb(0.95, 0.75, 0.45) };
    let c = if edge { Color::srgb(0.25, 0.12, 0.08) } else { base }.to_srgba();
    Color::srgb(c.red * ao, c.green * ao, c.blue * ao)
}
