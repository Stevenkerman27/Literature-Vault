# PDF Figure Extraction

`tools/extract-pdf-figure.ps1` renders selected PDF pages with Poppler and exports each page, or a fixed pixel crop from it, as a figure image. It is intended for adding paper figures to Obsidian notes.

## Requirements

- Windows PowerShell 5.1+ or PowerShell 7 on Windows.
- Poppler's `pdftoppm` available in `PATH`, or its executable path supplied through `-RendererPath`.
- The script uses `System.Drawing` for cropping and image conversion.

## Basic Usage

Export an entire page at 300 DPI:

```powershell
.\tools\extract-pdf-figure.ps1 `
  -InputPdf 'D:\3D\Projects\Papers\paper.pdf' `
  -Page 3 `
  -OutputDirectory '.\Literature\assets\paper-slug' `
  -NamePrefix 'overview'
```

Crop a figure from a page. `Crop` is `x, y, width, height` in pixels of the rasterized page, so coordinates depend on `Dpi`:

```powershell
.\tools\extract-pdf-figure.ps1 `
  -InputPdf 'D:\3D\Projects\Papers\paper.pdf' `
  -Page 3 `
  -Dpi 300 `
  -Crop 220,315,2100,650 `
  -OutputDirectory '.\Literature\assets\paper-slug' `
  -NamePrefix 'method-overview'
```

The second example creates `method-overview-p003.png`, which can be embedded from a note in `Literature/` as:

```markdown
![[assets/paper-slug/method-overview-p003.png]]
```

## Options

| Parameter | Purpose |
| --- | --- |
| `-Page 3,4,5` | Render one or more page numbers. The same crop is applied to every selected page. |
| `-Dpi 72..1200` | Rendering resolution. Use 300 for normal paper figures; increase it for small labels. |
| `-Crop x,y,width,height` | Optional rectangle in rendered-page pixels. Omit it to export the entire page. |
| `-ImageFormat Png` or `Jpeg` | Output format; PNG is the default and best for diagrams. |
| `-KeepRenderedPage` | Keep the uncropped rendered page alongside the figure. |
| `-RendererPath` | Full path to `pdftoppm.exe` when Poppler is not on `PATH`. |

## Workflow Notes

1. First export the full page with `-KeepRenderedPage` or without `-Crop` to inspect it.
2. Determine crop coordinates at the same DPI that will be used for the final export.
3. Run the crop command and visually check that labels, arrows, and panel markers are intact.
4. Commit only the final attachment under `Literature/assets/`; the script removes temporary render files automatically.

For the Zhu et al. 2017 note, Figure 2 was rendered at 300 DPI and cropped with `220,315,2100,650`.
