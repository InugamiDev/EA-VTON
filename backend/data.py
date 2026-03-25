"""Real UNIQLO garment data with verified CDN image URLs.

Product data sourced from UNIQLO US API (commerce/v5).
Image URLs from image.uniqlo.com CDN — verified accessible.
"""

GARMENTS: list[dict] = [
    {
        "id": "uniqlo-supima-crew-white",
        "name": "Supima Cotton Crew Neck T-Shirt",
        "type": "t-shirt",
        "brand": "UNIQLO",
        "description": (
            "Made from 100% SUPIMA cotton with long fibers that create a fine, "
            "smooth fabric surface. Clean appearance and consistent coloring "
            "compared to standard cotton."
        ),
        "price": 19.90,
        "colors": ["White", "Black", "Green"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "images": [
            {
                "id": "img-supima-white",
                "url": "https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/465759/item/goods_00_465759_3x4.jpg",
                "type": "display",
            },
            {
                "id": "img-supima-white-tryon",
                "url": "https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/465759/item/goods_00_465759_3x4.jpg",
                "type": "tryon_input",
            },
        ],
        "sizeChart": [
            {"size": "XS", "chestMin": 82, "chestMax": 88, "lengthCm": 64},
            {"size": "S", "chestMin": 88, "chestMax": 94, "lengthCm": 66},
            {"size": "M", "chestMin": 94, "chestMax": 100, "lengthCm": 69},
            {"size": "L", "chestMin": 100, "chestMax": 106, "lengthCm": 72},
            {"size": "XL", "chestMin": 106, "chestMax": 114, "lengthCm": 74},
        ],
    },
    {
        "id": "uniqlo-supima-crew-black",
        "name": "Supima Cotton Crew Neck T-Shirt",
        "type": "t-shirt",
        "brand": "UNIQLO",
        "description": (
            "Premium Supima cotton tee in black. Exceptionally soft with a "
            "clean crew neckline and relaxed fit for any occasion."
        ),
        "price": 19.90,
        "colors": ["Black", "White", "Navy", "Green"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "images": [
            {
                "id": "img-supima-black",
                "url": "https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/465759/item/goods_09_465759_3x4.jpg",
                "type": "display",
            },
            {
                "id": "img-supima-black-tryon",
                "url": "https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/465759/item/goods_09_465759_3x4.jpg",
                "type": "tryon_input",
            },
        ],
        "sizeChart": [
            {"size": "XS", "chestMin": 82, "chestMax": 88, "lengthCm": 64},
            {"size": "S", "chestMin": 88, "chestMax": 94, "lengthCm": 66},
            {"size": "M", "chestMin": 94, "chestMax": 100, "lengthCm": 69},
            {"size": "L", "chestMin": 100, "chestMax": 106, "lengthCm": 72},
            {"size": "XL", "chestMin": 106, "chestMax": 114, "lengthCm": 74},
        ],
    },
    {
        "id": "uniqlo-airism-vneck",
        "name": "AIRism Cotton V-Neck T-Shirt",
        "type": "t-shirt",
        "brand": "UNIQLO",
        "description": (
            "Comfortable AIRism fabric with the look of cotton. Accented by "
            "an overlapping V-neck design with meticulous stitching details."
        ),
        "price": 14.90,
        "colors": ["White", "Black", "Navy"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "images": [
            {
                "id": "img-airism-white",
                "url": "https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/478984/item/goods_00_478984_3x4.jpg",
                "type": "display",
            },
            {
                "id": "img-airism-white-tryon",
                "url": "https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/478984/item/goods_00_478984_3x4.jpg",
                "type": "tryon_input",
            },
        ],
        "sizeChart": [
            {"size": "XS", "chestMin": 82, "chestMax": 88, "lengthCm": 63},
            {"size": "S", "chestMin": 88, "chestMax": 94, "lengthCm": 65},
            {"size": "M", "chestMin": 94, "chestMax": 100, "lengthCm": 68},
            {"size": "L", "chestMin": 100, "chestMax": 106, "lengthCm": 71},
            {"size": "XL", "chestMin": 106, "chestMax": 114, "lengthCm": 73},
        ],
    },
    {
        "id": "uniqlo-oxford-slim-navy",
        "name": "Oxford Slim-Fit Long-Sleeve Shirt",
        "type": "shirt",
        "brand": "UNIQLO",
        "description": (
            "Finely woven premium Oxford fabric with a crisp, supple texture. "
            "Button-down collar and slim fit for a sleek look."
        ),
        "price": 29.90,
        "colors": ["Navy", "Off White", "Blue", "Pink"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "images": [
            {
                "id": "img-oxford-navy",
                "url": "https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/456630/item/goods_69_456630_3x4.jpg",
                "type": "display",
            },
            {
                "id": "img-oxford-navy-tryon",
                "url": "https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/456630/item/goods_69_456630_3x4.jpg",
                "type": "tryon_input",
            },
        ],
        "sizeChart": [
            {"size": "XS", "chestMin": 84, "chestMax": 90, "lengthCm": 68},
            {"size": "S", "chestMin": 90, "chestMax": 96, "lengthCm": 70},
            {"size": "M", "chestMin": 96, "chestMax": 102, "lengthCm": 73},
            {"size": "L", "chestMin": 102, "chestMax": 108, "lengthCm": 76},
            {"size": "XL", "chestMin": 108, "chestMax": 116, "lengthCm": 78},
        ],
    },
    {
        "id": "uniqlo-denim-boxy-shirt",
        "name": "Denim Boxy Shirt — Half Sleeve",
        "type": "shirt",
        "brand": "UNIQLO",
        "description": (
            "Relaxed boxy silhouette in soft denim with a half-sleeve cut. "
            "Casual enough for weekends, polished enough for the office."
        ),
        "price": 39.90,
        "colors": ["Natural", "Black", "Navy", "Blue"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "images": [
            {
                "id": "img-denim-natural",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/483877/item/usgoods_30_483877_3x4.jpg",
                "type": "display",
            },
            {
                "id": "img-denim-natural-tryon",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/483877/item/usgoods_30_483877_3x4.jpg",
                "type": "tryon_input",
            },
        ],
        "sizeChart": [
            {"size": "XS", "chestMin": 88, "chestMax": 94, "lengthCm": 66},
            {"size": "S", "chestMin": 94, "chestMax": 100, "lengthCm": 68},
            {"size": "M", "chestMin": 100, "chestMax": 106, "lengthCm": 71},
            {"size": "L", "chestMin": 106, "chestMax": 114, "lengthCm": 74},
            {"size": "XL", "chestMin": 114, "chestMax": 122, "lengthCm": 76},
        ],
    },
    {
        "id": "uniqlo-sweat-hoodie",
        "name": "Sweat Pullover Hoodie",
        "type": "hoodie",
        "brand": "UNIQLO",
        "description": (
            "Classic pullover hoodie in a comfortable sweat fabric. "
            "Kangaroo pocket and adjustable drawstring hood for everyday wear."
        ),
        "price": 39.90,
        "colors": ["Olive", "Off White", "Gray", "Black"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "images": [
            {
                "id": "img-hoodie-olive",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/475378/item/usgoods_55_475378_3x4.jpg",
                "type": "display",
            },
            {
                "id": "img-hoodie-olive-tryon",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/475378/item/usgoods_55_475378_3x4.jpg",
                "type": "tryon_input",
            },
        ],
        "sizeChart": [
            {"size": "XS", "chestMin": 88, "chestMax": 94, "lengthCm": 63},
            {"size": "S", "chestMin": 94, "chestMax": 100, "lengthCm": 65},
            {"size": "M", "chestMin": 100, "chestMax": 106, "lengthCm": 68},
            {"size": "L", "chestMin": 106, "chestMax": 114, "lengthCm": 71},
            {"size": "XL", "chestMin": 114, "chestMax": 122, "lengthCm": 73},
        ],
    },
    {
        "id": "uniqlo-oversized-hoodie",
        "name": "Sweat Oversized Pullover Hoodie",
        "type": "hoodie",
        "brand": "UNIQLO",
        "description": (
            "Relaxed oversized silhouette with dropped shoulders. "
            "Heavyweight sweat fabric for a premium feel and structure."
        ),
        "price": 49.90,
        "colors": ["Red", "Gray", "Black", "Beige"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "images": [
            {
                "id": "img-oversized-hoodie-red",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/471808/item/usgoods_17_471808_3x4.jpg",
                "type": "display",
            },
            {
                "id": "img-oversized-hoodie-red-tryon",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/471808/item/usgoods_17_471808_3x4.jpg",
                "type": "tryon_input",
            },
        ],
        "sizeChart": [
            {"size": "XS", "chestMin": 96, "chestMax": 104, "lengthCm": 67},
            {"size": "S", "chestMin": 104, "chestMax": 112, "lengthCm": 69},
            {"size": "M", "chestMin": 112, "chestMax": 120, "lengthCm": 71},
            {"size": "L", "chestMin": 120, "chestMax": 128, "lengthCm": 73},
            {"size": "XL", "chestMin": 128, "chestMax": 136, "lengthCm": 75},
        ],
    },
    {
        "id": "uniqlo-uv-parka",
        "name": "Pocketable UV Protection Parka",
        "type": "jacket",
        "brand": "UNIQLO",
        "description": (
            "Ultra-light parka with UV protection that folds into its own pocket. "
            "Water-repellent finish keeps you dry in light rain."
        ),
        "price": 49.90,
        "colors": ["Pink", "White", "Black", "Gray", "Green"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "images": [
            {
                "id": "img-parka-pink",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/485671/item/usgoods_10_485671_3x4.jpg",
                "type": "display",
            },
            {
                "id": "img-parka-pink-tryon",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/485671/item/usgoods_10_485671_3x4.jpg",
                "type": "tryon_input",
            },
        ],
        "sizeChart": [
            {"size": "XS", "chestMin": 86, "chestMax": 92, "lengthCm": 63},
            {"size": "S", "chestMin": 92, "chestMax": 98, "lengthCm": 66},
            {"size": "M", "chestMin": 98, "chestMax": 104, "lengthCm": 69},
            {"size": "L", "chestMin": 104, "chestMax": 112, "lengthCm": 72},
            {"size": "XL", "chestMin": 112, "chestMax": 120, "lengthCm": 74},
        ],
    },
    {
        "id": "uniqlo-windproof-parka",
        "name": "Windproof Parka",
        "type": "jacket",
        "brand": "UNIQLO",
        "description": (
            "Windproof outer layer with DWR water-repellent finish. "
            "Lightweight layering piece with adjustable hood and cuffs."
        ),
        "price": 49.90,
        "colors": ["Brown", "Black", "Navy", "Olive"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "images": [
            {
                "id": "img-windproof-brown",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/478231/item/usgoods_34_478231_3x4.jpg",
                "type": "display",
            },
            {
                "id": "img-windproof-brown-tryon",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/478231/item/usgoods_34_478231_3x4.jpg",
                "type": "tryon_input",
            },
        ],
        "sizeChart": [
            {"size": "XS", "chestMin": 86, "chestMax": 92, "lengthCm": 64},
            {"size": "S", "chestMin": 92, "chestMax": 98, "lengthCm": 67},
            {"size": "M", "chestMin": 98, "chestMax": 104, "lengthCm": 70},
            {"size": "L", "chestMin": 104, "chestMax": 112, "lengthCm": 73},
            {"size": "XL", "chestMin": 112, "chestMax": 120, "lengthCm": 75},
        ],
    },
    {
        "id": "uniqlo-cotton-polo-sweater",
        "name": "Smooth Cotton Polo Sweater",
        "type": "polo",
        "brand": "UNIQLO",
        "description": (
            "Knitted polo in smooth cotton with a refined sheen. "
            "Ribbed collar and placket for a smart-casual look."
        ),
        "price": 49.90,
        "colors": ["Yellow", "Gray", "Black"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "images": [
            {
                "id": "img-polo-yellow",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/482971/item/usgoods_41_482971_3x4.jpg",
                "type": "display",
            },
            {
                "id": "img-polo-yellow-tryon",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/482971/item/usgoods_41_482971_3x4.jpg",
                "type": "tryon_input",
            },
        ],
        "sizeChart": [
            {"size": "XS", "chestMin": 84, "chestMax": 90, "lengthCm": 63},
            {"size": "S", "chestMin": 90, "chestMax": 96, "lengthCm": 66},
            {"size": "M", "chestMin": 96, "chestMax": 102, "lengthCm": 69},
            {"size": "L", "chestMin": 102, "chestMax": 110, "lengthCm": 72},
            {"size": "XL", "chestMin": 110, "chestMax": 118, "lengthCm": 74},
        ],
    },
    {
        "id": "uniqlo-knitted-skipper-polo",
        "name": "Washable Knitted Skipper Polo",
        "type": "polo",
        "brand": "UNIQLO",
        "description": (
            "Machine-washable knitted polo with a skipper collar. "
            "Easy-care blend that keeps its shape wash after wash."
        ),
        "price": 39.90,
        "colors": ["Natural", "Light Brown", "Navy", "Dark Olive"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "images": [
            {
                "id": "img-skipper-natural",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/482328/item/usgoods_30_482328_3x4.jpg",
                "type": "display",
            },
            {
                "id": "img-skipper-natural-tryon",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/482328/item/usgoods_30_482328_3x4.jpg",
                "type": "tryon_input",
            },
        ],
        "sizeChart": [
            {"size": "XS", "chestMin": 84, "chestMax": 90, "lengthCm": 63},
            {"size": "S", "chestMin": 90, "chestMax": 96, "lengthCm": 66},
            {"size": "M", "chestMin": 96, "chestMax": 102, "lengthCm": 69},
            {"size": "L", "chestMin": 102, "chestMax": 110, "lengthCm": 72},
            {"size": "XL", "chestMin": 110, "chestMax": 118, "lengthCm": 74},
        ],
    },
    {
        "id": "uniqlo-merino-sweater",
        "name": "Extra Fine Merino Crew Neck Sweater",
        "type": "sweater",
        "brand": "UNIQLO",
        "description": (
            "Extra fine merino wool for a smooth, luxurious feel. "
            "Lightweight enough for layering yet warm enough on its own."
        ),
        "price": 39.90,
        "colors": ["Olive", "Gray", "Black"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "images": [
            {
                "id": "img-merino-olive",
                "url": "https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/450535/item/goods_57_450535_3x4.jpg",
                "type": "display",
            },
            {
                "id": "img-merino-olive-tryon",
                "url": "https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/450535/item/goods_57_450535_3x4.jpg",
                "type": "tryon_input",
            },
        ],
        "sizeChart": [
            {"size": "XS", "chestMin": 84, "chestMax": 90, "lengthCm": 62},
            {"size": "S", "chestMin": 90, "chestMax": 96, "lengthCm": 65},
            {"size": "M", "chestMin": 96, "chestMax": 102, "lengthCm": 68},
            {"size": "L", "chestMin": 102, "chestMax": 110, "lengthCm": 71},
            {"size": "XL", "chestMin": 110, "chestMax": 118, "lengthCm": 73},
        ],
    },
    {
        "id": "uniqlo-pufftech-vest",
        "name": "PUFFTECH Washable Vest",
        "type": "vest",
        "brand": "UNIQLO",
        "description": (
            "Lightweight puffer vest with PUFFTECH insulation. "
            "Machine washable, packable, and warm enough for transitional weather."
        ),
        "price": 29.90,
        "colors": ["Blue", "Black", "Off White", "Gray", "Olive"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "images": [
            {
                "id": "img-pufftech-blue",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/478573/item/usgoods_67_478573_3x4.jpg",
                "type": "display",
            },
            {
                "id": "img-pufftech-blue-tryon",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/478573/item/usgoods_67_478573_3x4.jpg",
                "type": "tryon_input",
            },
        ],
        "sizeChart": [
            {"size": "XS", "chestMin": 86, "chestMax": 92, "lengthCm": 60},
            {"size": "S", "chestMin": 92, "chestMax": 98, "lengthCm": 63},
            {"size": "M", "chestMin": 98, "chestMax": 104, "lengthCm": 66},
            {"size": "L", "chestMin": 104, "chestMax": 112, "lengthCm": 69},
            {"size": "XL", "chestMin": 112, "chestMax": 120, "lengthCm": 71},
        ],
    },
    {
        "id": "uniqlo-zip-hoodie",
        "name": "Sweat Oversized Full-Zip Hoodie",
        "type": "hoodie",
        "brand": "UNIQLO",
        "description": (
            "Oversized zip-up hoodie with a heavyweight fabric for structure. "
            "Two-way zipper and side pockets for versatile layering."
        ),
        "price": 59.90,
        "colors": ["Beige", "Black", "Gray", "Navy"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "images": [
            {
                "id": "img-zip-hoodie-beige",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/485735/item/usgoods_30_485735_3x4.jpg",
                "type": "display",
            },
            {
                "id": "img-zip-hoodie-beige-tryon",
                "url": "https://image.uniqlo.com/UQ/ST3/us/imagesgoods/485735/item/usgoods_30_485735_3x4.jpg",
                "type": "tryon_input",
            },
        ],
        "sizeChart": [
            {"size": "XS", "chestMin": 96, "chestMax": 104, "lengthCm": 65},
            {"size": "S", "chestMin": 104, "chestMax": 112, "lengthCm": 68},
            {"size": "M", "chestMin": 112, "chestMax": 120, "lengthCm": 71},
            {"size": "L", "chestMin": 120, "chestMax": 128, "lengthCm": 73},
            {"size": "XL", "chestMin": 128, "chestMax": 136, "lengthCm": 75},
        ],
    },
]


def get_garment_by_id(garment_id: str) -> dict | None:
    """Look up a garment by its ID."""
    for g in GARMENTS:
        if g["id"] == garment_id:
            return g
    return None
