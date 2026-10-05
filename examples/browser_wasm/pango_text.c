#include <emscripten.h>
#include <cairo/cairo.h>
#include <cairo/cairo-svg.h>
#include <pango/pangocairo.h>
#include <glib.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static unsigned long manim_svg_counter = 0;

static char *read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }

    long size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }

    char *data = (char *)malloc((size_t)size + 1);
    if (!data) {
        fclose(file);
        return NULL;
    }

    size_t read = fread(data, 1, (size_t)size, file);
    fclose(file);

    if (read != (size_t)size) {
        free(data);
        return NULL;
    }

    data[size] = '\0';
    return data;
}

EMSCRIPTEN_KEEPALIVE
char *manim_pango_text_to_svg(
    const char *markup,
    int justify,
    double indent,
    int alignment,
    double width
) {
    if (!markup) return NULL;

    char path[128];
    unsigned long id = __atomic_fetch_add(&manim_svg_counter, 1, __ATOMIC_RELAXED);
    snprintf(path, sizeof(path), "/tmp/manim-pango-%lu.svg", id);

    cairo_surface_t *surface = cairo_svg_surface_create(
        path, 16384.0, 16384.0
    );
    if (cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS) {
        cairo_surface_destroy(surface);
        return NULL;
    }

    cairo_svg_surface_set_document_unit(surface, CAIRO_SVG_UNIT_PX);
    cairo_svg_surface_restrict_to_version(surface, CAIRO_SVG_VERSION_1_2);
    cairo_surface_set_device_scale(surface, 1.0, 1.0);

    cairo_t *cr = cairo_create(surface);
    if (cairo_status(cr) != CAIRO_STATUS_SUCCESS) {
        cairo_destroy(cr);
        cairo_surface_destroy(surface);
        remove(path);
        return NULL;
    }

    PangoLayout *layout = pango_cairo_create_layout(cr);
    if (!layout) {
        cairo_destroy(cr);
        cairo_surface_destroy(surface);
        remove(path);
        return NULL;
    }

    pango_layout_set_markup(layout, markup, -1);

    PangoFontDescription *font = pango_font_description_from_string("sans 48");
    pango_layout_set_font_description(layout, font);
    pango_font_description_free(font);

    if (width > 0) {
        pango_layout_set_width(layout, (int)(width * PANGO_SCALE));
    } else {
        pango_layout_set_width(layout, -1);
    }

    pango_layout_set_justify(layout, justify != 0);
    pango_layout_set_indent(layout, (int)(indent * PANGO_SCALE));

    PangoAlignment pango_alignment = PANGO_ALIGN_CENTER;
    if (alignment == 0) {
        pango_alignment = PANGO_ALIGN_LEFT;
    } else if (alignment == 2) {
        pango_alignment = PANGO_ALIGN_RIGHT;
    }
    pango_layout_set_alignment(layout, pango_alignment);

    pango_cairo_show_layout(cr, layout);

    g_object_unref(layout);
    cairo_destroy(cr);
    cairo_surface_destroy(surface);

    char *data = read_file(path);
    remove(path);
    return data;
}

EMSCRIPTEN_KEEPALIVE
void manim_pango_free(char *data) {
    free(data);
}
