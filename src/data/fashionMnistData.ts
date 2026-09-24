/**
 * Fashion-MNIST Dataset Definitions and Embedded Samples
 * Classes:
 * 0: T-shirt/top (Camiseta)
 * 1: Trouser (Pantalón)
 * 2: Pullover (Pulóver)
 * 3: Dress (Vestido)
 * 4: Coat (Abrigo)
 * 5: Sandal (Sandalia)
 * 6: Shirt (Camisa)
 * 7: Sneaker (Zapatilla deportiva)
 * 8: Bag (Bolso)
 * 9: Ankle boot (Botín)
 */

export interface FashionClass {
  id: number;
  name: string;
  nameEn: string;
  category: 'vestimenta' | 'calzado' | 'accesorio';
  description: string;
  features: string[];
}

export const FASHION_CLASSES: FashionClass[] = [
  {
    id: 0,
    name: 'Camiseta (T-shirt/top)',
    nameEn: 'T-shirt/top',
    category: 'vestimenta',
    description: 'Prenda superior de manga corta y cuello redondo o en V.',
    features: ['Mangas cortas simétricas', 'Cuello despejado', 'Cuerpo rectangular']
  },
  {
    id: 1,
    name: 'Pantalón (Trouser)',
    nameEn: 'Trouser',
    category: 'vestimenta',
    description: 'Prenda para extremidades inferiores con dos perneras alargadas.',
    features: ['Dos columnas verticales paralelas', 'Cintura superior', 'Hueco central entre perneras']
  },
  {
    id: 2,
    name: 'Pulóver (Pullover)',
    nameEn: 'Pullover',
    category: 'vestimenta',
    description: 'Suéter de tejido denso con mangas largas y dobladillo cerrado.',
    features: ['Mangas largas completas', 'Silueta más ancha y tupida', 'Cuello cerrado']
  },
  {
    id: 3,
    name: 'Vestido (Dress)',
    nameEn: 'Dress',
    category: 'vestimenta',
    description: 'Prenda continua con torso ajustado y falda que se ensancha hacia abajo.',
    features: ['Cintura estrecha', 'Falda acampanada inferior', 'Continuidad de una sola pieza']
  },
  {
    id: 4,
    name: 'Abrigo (Coat)',
    nameEn: 'Coat',
    category: 'vestimenta',
    description: 'Prenda exterior larga y pesada, generalmente con solapa o botones.',
    features: ['Longitud mayor que camisa', 'Mangas largas robustas', 'Línea de apertura frontal']
  },
  {
    id: 5,
    name: 'Sandalia (Sandal)',
    nameEn: 'Sandal',
    category: 'calzado',
    description: 'Calzado ligero y abierto con tiras delgadas y base plana.',
    features: ['Espacios vacíos en empeine', 'Tiras finas de sujeción', 'Suela delgada']
  },
  {
    id: 6,
    name: 'Camisa (Shirt)',
    nameEn: 'Shirt',
    category: 'vestimenta',
    description: 'Prenda superior con cuello camisero, botones frontales y mangas.',
    features: ['Cuello rígido en pico', 'Solapa central de botones', 'Mangas medias o largas']
  },
  {
    id: 7,
    name: 'Zapatilla (Sneaker)',
    nameEn: 'Sneaker',
    category: 'calzado',
    description: 'Calzado deportivo con suela de goma amortiguada y empeine curvado.',
    features: ['Puntera redondeada', 'Suela gruesa horizontal', 'Curva ergonómica del empeine']
  },
  {
    id: 8,
    name: 'Bolso (Bag)',
    nameEn: 'Bag',
    category: 'accesorio',
    description: 'Accesorio contenedor con asas superiores y cuerpo geométrico.',
    features: ['Asas delgadas arqueadas', 'Cuerpo compacto cuadrangular o trapezoidal']
  },
  {
    id: 9,
    name: 'Botín (Ankle boot)',
    nameEn: 'Ankle boot',
    category: 'calzado',
    description: 'Calzado alto que cubre el tobillo, con caña vertical y suela sólida.',
    features: ['Caña vertical que cubre el tobillo', 'Puntera orientada', 'Tacón o suela elevada']
  }
];

export interface FashionSample {
  id: string;
  classId: number;
  label: string;
  // 28x28 normalized array (784 floats between 0 and 1)
  data: number[];
}

/**
 * Procedural rasterizer that creates authentic Fashion-MNIST style 28x28 grayscale images
 * with natural edge anti-aliasing and variation.
 */
function createGrid28(fn: (x: number, y: number) => number): number[] {
  const result: number[] = new Array(784).fill(0);
  for (let r = 0; r < 28; r++) {
    for (let c = 0; c < 28; c++) {
      // Normal coordinates: x in [-1, 1], y in [-1, 1]
      const nx = (c - 13.5) / 13.5;
      const ny = (r - 13.5) / 13.5;
      const val = Math.max(0, Math.min(1, fn(nx, ny)));
      result[r * 28 + c] = Math.round(val * 100) / 100;
    }
  }
  return result;
}

// Generate canonical prototypes for each Fashion-MNIST category
export function generateFashionSamples(): FashionSample[] {
  const samples: FashionSample[] = [];

  // Helper for Gaussian blur / soft distance
  const softStep = (val: number, edge: number, width: number) => {
    return Math.max(0, Math.min(1, (edge + width - val) / (2 * width)));
  };

  // 0: T-shirt / top
  // Variant A: Standard Crewneck T-shirt
  samples.push({
    id: 'tshirt-1',
    classId: 0,
    label: 'Camiseta Básica',
    data: createGrid28((x, y) => {
      // Neck cutout
      if (y < -0.45 && Math.abs(x) < 0.28) return 0;
      // Torso
      if (y >= -0.45 && y <= 0.75 && Math.abs(x) <= 0.52) return 0.9;
      // Short sleeves
      if (y >= -0.65 && y <= -0.15 && Math.abs(x) <= 0.88 && Math.abs(x) >= 0.45) {
        const sleeveSlope = (Math.abs(x) - 0.45) * 0.7;
        if (y <= -0.55 + sleeveSlope && y >= -0.7 + sleeveSlope) return 0.85;
        if (y >= -0.65 + sleeveSlope && y <= -0.15 + sleeveSlope) return 0.85;
      }
      return 0;
    })
  });

  // Variant B: V-Neck T-shirt
  samples.push({
    id: 'tshirt-2',
    classId: 0,
    label: 'Camiseta Cuello V',
    data: createGrid28((x, y) => {
      // V-neck cutout
      if (y < -0.2 && y > -0.75 && Math.abs(x) < (y + 0.75) * 0.5) return 0;
      // Torso
      if (y >= -0.45 && y <= 0.8 && Math.abs(x) <= 0.48) return 0.92;
      // Sleeves
      if (y >= -0.6 && y <= -0.1 && Math.abs(x) <= 0.82 && Math.abs(x) >= 0.42) return 0.8;
      return 0;
    })
  });

  // 1: Trouser
  // Variant A: Straight Trouser
  samples.push({
    id: 'trouser-1',
    classId: 1,
    label: 'Pantalón Recto',
    data: createGrid28((x, y) => {
      // Waistband
      if (y >= -0.75 && y <= -0.45 && Math.abs(x) <= 0.55) return 0.95;
      // Legs
      if (y > -0.45 && y <= 0.85) {
        // Crotch split
        if (Math.abs(x) < 0.12 && y > -0.25) return 0;
        if (Math.abs(x) >= 0.08 && Math.abs(x) <= 0.5) return 0.9;
      }
      return 0;
    })
  });

  // Variant B: Slim Fit Trouser
  samples.push({
    id: 'trouser-2',
    classId: 1,
    label: 'Pantalón Slim',
    data: createGrid28((x, y) => {
      if (y >= -0.8 && y <= -0.45 && Math.abs(x) <= 0.5) return 0.95;
      if (y > -0.45 && y <= 0.88) {
        if (Math.abs(x) < 0.14 && y > -0.28) return 0;
        const taper = 0.45 - (y + 0.45) * 0.12;
        if (Math.abs(x) >= 0.08 && Math.abs(x) <= taper) return 0.88;
      }
      return 0;
    })
  });

  // 2: Pullover
  // Variant A: Thick Sweater
  samples.push({
    id: 'pullover-1',
    classId: 2,
    label: 'Pulóver Clásico',
    data: createGrid28((x, y) => {
      // High neck collar
      if (y >= -0.75 && y <= -0.55 && Math.abs(x) <= 0.25) return 0.88;
      // Bulkier Torso
      if (y >= -0.55 && y <= 0.72 && Math.abs(x) <= 0.55) return 0.92;
      // Long sleeves down to wrists
      if (y >= -0.65 && y <= 0.6) {
        const outerBound = 0.88 - (y + 0.65) * 0.18;
        const innerBound = Math.max(0.48, 0.7 - (y + 0.65) * 0.2);
        if (Math.abs(x) <= outerBound && Math.abs(x) >= innerBound) return 0.85;
      }
      return 0;
    })
  });

  // 3: Dress
  // Variant A: A-line Dress
  samples.push({
    id: 'dress-1',
    classId: 3,
    label: 'Vestido Acampanado',
    data: createGrid28((x, y) => {
      // Straps / top
      if (y >= -0.75 && y <= -0.4) {
        if (Math.abs(x) <= 0.35 && Math.abs(x) >= 0.12) return 0.9;
        if (y >= -0.55 && Math.abs(x) < 0.12) return 0.9;
      }
      // Waist
      if (y > -0.4 && y <= -0.05 && Math.abs(x) <= 0.32) return 0.95;
      // Flared skirt
      if (y > -0.05 && y <= 0.82) {
        const skirtWidth = 0.32 + (y + 0.05) * 0.55;
        if (Math.abs(x) <= skirtWidth) return 0.9;
      }
      return 0;
    })
  });

  // 4: Coat
  // Variant A: Overcoat
  samples.push({
    id: 'coat-1',
    classId: 4,
    label: 'Abrigo Largo',
    data: createGrid28((x, y) => {
      // High collar & lapel
      if (y >= -0.8 && y <= -0.55 && Math.abs(x) <= 0.35) return 0.9;
      // Long body almost to knees
      if (y >= -0.55 && y <= 0.88 && Math.abs(x) <= 0.6) {
        // Vertical button/slit seam
        if (Math.abs(x) < 0.04 && y > -0.4) return 0.35;
        return 0.92;
      }
      // Thick sleeves
      if (y >= -0.65 && y <= 0.65 && Math.abs(x) <= 0.92 && Math.abs(x) >= 0.55) return 0.88;
      return 0;
    })
  });

  // 5: Sandal
  // Variant A: Flat Strap Sandal
  samples.push({
    id: 'sandal-1',
    classId: 5,
    label: 'Sandalia Abierta',
    data: createGrid28((x, y) => {
      // Thin flat sole
      if (y >= 0.45 && y <= 0.68 && x >= -0.75 && x <= 0.75) return 0.95;
      // Ankle strap
      if (x >= -0.55 && x <= -0.35 && y >= -0.15 && y <= 0.5) return 0.85;
      // Cross strap
      if (Math.abs((y - 0.2) + 0.5 * (x - 0.1)) < 0.12 && x >= -0.2 && x <= 0.6) return 0.8;
      // Toe loop
      if (x >= 0.45 && x <= 0.65 && y >= 0.25 && y <= 0.48) return 0.85;
      return 0;
    })
  });

  // 6: Shirt
  // Variant A: Button-up Collared Shirt
  samples.push({
    id: 'shirt-1',
    classId: 6,
    label: 'Camisa Formal',
    data: createGrid28((x, y) => {
      // Collar wings
      if (y >= -0.78 && y <= -0.52 && Math.abs(x) <= 0.4) {
        if (Math.abs(x) < 0.08) return 0.2; // center opening
        return 0.95;
      }
      // Torso
      if (y >= -0.52 && y <= 0.75 && Math.abs(x) <= 0.5) {
        // Placket line
        if (Math.abs(x) < 0.035) return 0.4;
        return 0.9;
      }
      // Medium sleeves
      if (y >= -0.6 && y <= 0.35 && Math.abs(x) <= 0.84 && Math.abs(x) >= 0.46) return 0.82;
      return 0;
    })
  });

  // 7: Sneaker
  // Variant A: Low-top athletic sneaker
  samples.push({
    id: 'sneaker-1',
    classId: 7,
    label: 'Zapatilla Running',
    data: createGrid28((x, y) => {
      // Curved thick sole
      if (y >= 0.35 && y <= 0.65 && x >= -0.85 && x <= 0.85) {
        if (x > 0.6) {
          // Upturned toe
          if (y >= 0.28 && y <= 0.55) return 0.95;
        }
        return 0.95;
      }
      // Body of shoe
      if (x >= -0.8 && x <= 0.75 && y >= 0.0 && y <= 0.4) {
        // Tongue & laces dip
        if (x >= -0.1 && x <= 0.35 && y <= 0.12) return 0.5;
        // Collar / heel cup
        if (x <= -0.4 && y <= -0.1) return 0.9;
        return 0.88;
      }
      return 0;
    })
  });

  // Variant B: Retro Sneaker
  samples.push({
    id: 'sneaker-2',
    classId: 7,
    label: 'Zapatilla Urbana',
    data: createGrid28((x, y) => {
      if (y >= 0.38 && y <= 0.68 && x >= -0.82 && x <= 0.82) return 0.95;
      if (x >= -0.78 && x <= 0.72 && y >= 0.05 && y <= 0.42) return 0.9;
      if (x >= -0.75 && x <= -0.35 && y >= -0.15 && y <= 0.1) return 0.85;
      return 0;
    })
  });

  // 8: Bag
  // Variant A: Tote Bag
  samples.push({
    id: 'bag-1',
    classId: 8,
    label: 'Bolso Tote',
    data: createGrid28((x, y) => {
      // Thin strap handles
      if (y >= -0.85 && y <= -0.15) {
        if ((Math.abs(x - 0.25) < 0.08 || Math.abs(x + 0.25) < 0.08) && y >= -0.8) {
          if (Math.abs(x) < 0.35) return 0.85;
        }
      }
      // Bag body (rectangle)
      if (y >= -0.15 && y <= 0.75 && Math.abs(x) <= 0.6) return 0.92;
      return 0;
    })
  });

  // Variant B: Handbag
  samples.push({
    id: 'bag-2',
    classId: 8,
    label: 'Bolso de Mano',
    data: createGrid28((x, y) => {
      // Curved handle arch
      const distFromCenter = Math.sqrt(x * x + (y + 0.15) * (y + 0.15));
      if (y <= -0.15 && Math.abs(distFromCenter - 0.45) < 0.09) return 0.88;
      // Trapezoid body
      if (y >= -0.15 && y <= 0.72) {
        const halfWidth = 0.48 + (y + 0.15) * 0.15;
        if (Math.abs(x) <= halfWidth) return 0.94;
      }
      return 0;
    })
  });

  // 9: Ankle boot
  // Variant A: High ankle boot
  samples.push({
    id: 'boot-1',
    classId: 9,
    label: 'Botín Chelsea',
    data: createGrid28((x, y) => {
      // Sturdy sole with slight heel at back
      if (y >= 0.42 && y <= 0.7 && x >= -0.8 && x <= 0.82) {
        if (x <= -0.3 && y >= 0.55) return 0.98; // heel block
        return 0.92;
      }
      // Toe box
      if (x >= -0.1 && x <= 0.8 && y >= 0.15 && y <= 0.45) return 0.9;
      // High Ankle Shaft
      if (x >= -0.75 && x <= -0.05 && y >= -0.65 && y <= 0.45) return 0.92;
      return 0;
    })
  });

  // Variant B: Combat Boot
  samples.push({
    id: 'boot-2',
    classId: 9,
    label: 'Bota Militar',
    data: createGrid28((x, y) => {
      // Rugged lug sole
      if (y >= 0.45 && y <= 0.75 && x >= -0.82 && x <= 0.84) return 0.98;
      // Foot
      if (x >= -0.2 && x <= 0.82 && y >= 0.15 && y <= 0.48) return 0.92;
      // Tall shaft
      if (x >= -0.78 && x <= -0.08 && y >= -0.75 && y <= 0.48) return 0.92;
      return 0;
    })
  });

  return samples;
}

/**
 * Creates augmented batches for interactive training in browser
 */
export function generateTrainingBatch(size: number = 16): FashionSample[] {
  const baseSamples = generateFashionSamples();
  const batch: FashionSample[] = [];

  for (let i = 0; i < size; i++) {
    const randomClass = Math.floor(Math.random() * 10);
    const available = baseSamples.filter(s => s.classId === randomClass);
    const prototype = available[Math.floor(Math.random() * available.length)] || baseSamples[0];
    
    // Add realistic sensor noise, brightness jitter and tiny shifts
    const jitter = 0.85 + Math.random() * 0.3;
    const noiseLevel = 0.04;
    const shiftX = Math.floor(Math.random() * 3) - 1; // -1, 0, 1
    const shiftY = Math.floor(Math.random() * 3) - 1;

    const augmentedData = new Array(784).fill(0);
    for (let r = 0; r < 28; r++) {
      for (let c = 0; c < 28; c++) {
        const srcR = r - shiftY;
        const srcC = c - shiftX;
        let val = 0;
        if (srcR >= 0 && srcR < 28 && srcC >= 0 && srcC < 28) {
          val = prototype.data[srcR * 28 + srcC];
        }
        val = val * jitter + (Math.random() - 0.5) * noiseLevel;
        augmentedData[r * 28 + c] = Math.max(0, Math.min(1, Math.round(val * 100) / 100));
      }
    }

    batch.push({
      id: `train-${Date.now()}-${i}`,
      classId: randomClass,
      label: FASHION_CLASSES[randomClass].name,
      data: augmentedData
    });
  }

  return batch;
}
