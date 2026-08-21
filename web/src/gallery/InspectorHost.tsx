/**
 * Chooses which inspector the gallery shows.
 *
 * Doc 09: one item, or aggregate stats plus batch tools for a selection.
 * The choice is here rather than inside either panel so neither has to
 * know the other exists.
 */

import type { Item } from "~/api/types";
import { ItemInspector, SelectionInspector } from "./GalleryInspector";

interface Props {
  dataset: string;
  selectedItems: Item[];
  focusedItem: Item | undefined;
  onCaption(stem: string, caption: string): Promise<void>;
  onQuality(stem: string, quality: string | null): void;
  onQualityAll(quality: string | null): void;
  onAppend(text: string): Promise<void>;
  onClear(): void;
  onOpen(stem: string): void;
}

export function GalleryInspectorHost({
  dataset,
  selectedItems,
  focusedItem,
  onCaption,
  onQuality,
  onQualityAll,
  onAppend,
  onClear,
  onOpen,
}: Props) {
  if (selectedItems.length > 1) {
    return (
      <SelectionInspector
        items={selectedItems}
        onAppend={onAppend}
        onQualityAll={onQualityAll}
        onClear={onClear}
      />
    );
  }

  const item = focusedItem ?? selectedItems[0];
  if (!item) {
    return (
      <aside class="inspector" aria-label="Item">
        <div class="empty">
          <div class="empty__title">Nothing selected</div>
          <div>Click a thumbnail to inspect it</div>
        </div>
      </aside>
    );
  }

  return (
    <ItemInspector
      dataset={dataset}
      item={item}
      onCaption={onCaption}
      onQuality={onQuality}
      onOpen={onOpen}
    />
  );
}
