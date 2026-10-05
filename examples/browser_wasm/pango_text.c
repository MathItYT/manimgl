#include <emscripten.h>
#include <cairo/cairo.h>
#include <pango/pangocairo.h>
#include <glib.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *data;
    size_t size;
    size_t capacity;
} SvgBuffer;

static cairo_status_t write_svg(void *closure, const unsigned char *data, unsigned int length) {
    SvgBuffer *buffer = (SvgBuffer *)closure;
    size_t required = buffer->size + length + 1;
    if (required > buffer->capacity) {
        size_t capacity = buffer->capacity ? buffer->capacity : 4096;
        while (capacity < required) capacity *= 2;
        char *next = (char *)realloc(buffer->data, capacity);
        if (!next) return CAIRO_STATUS_NO_MEMORY;
        buffer->data = next;
        buffer->capacity = capacity;
    }
    memcpy(buffer->data + buffer->size, data, length);
    buffer->size += length;
    buffer->data[buffer->size] = '\0';
    return CAIRO_STATUS_SUCCESS;
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

    SvgBuffer buffer = {0};
    cairo_surface_t *surface = cairo_svg_surface_create_for_stream(
        write_svg, &buffer, 1.0, 1.0
    );
    cairo_t *cr = cairo_create(surface);
    PangoLayout *layout = pango_cairo_create_layout(cr);

    pango_layout_set_markup(layout, markup, -1);

    PangoFontDescription *font = pango_font_description_from_string(
        "sans 48"
    );
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
    if (alignment == 0) pango_alignment = PANGO_ALIGN_LEFT;
    else if (alignment == 2) pango_alignment = PANGO_ALIGN_RIGHT;
    pango_layout_set_alignment(layout, pango_alignment);

    int pixel_width = 0;
    int pixel_height = 0;
    pango_layout_get_pixel_size(layout, &pixel_width, &pixel_height);
    pixel_width = pixel_width > 0 ? pixel_width : 1;
    pixel_height = pixel_height > 0 ? pixel_height : 1;

    cairo_svg_surface_set_document_unit(surface, CAIRO_SVG_UNIT_PX);
    cairo_svg_surface_restrict_to_version(surface, CAIRO_SVG_VERSION_1_2);
    cairo_surface_set_device_scale(surface, 1.0, 1.0);

    cairo_destroy(cr);
    cairo_surface_destroy(surface);
    g_object_unref(layout);

    return buffer.data;
}

EMSCRIPTEN_KEEPALIVE
void manim_pango_free(char *data) {
    free(data);
}
