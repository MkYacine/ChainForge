import React, {
  useState,
  useRef,
  forwardRef,
  useImperativeHandle,
  useCallback,
} from "react";
import {
  Menu,
  Button,
  Card,
  Group,
  Text,
  ActionIcon,
  Modal,
  Switch,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { IconPlus, IconTrash, IconSettings } from "@tabler/icons-react";
import Form from "@rjsf/core";
import validator from "@rjsf/validator-ajv8";
import { v4 as uuid } from "uuid";
import { VectorstoreSchemas } from "./VectorstoreSchemas";

const vectorstoreTypes = [
  {
    value: "pinecone",
    label: "Pinecone",
    emoji: "☁️",
  },
  {
    value: "faiss",
    label: "FAISS",
    emoji: "💾",
  },
  {
    value: "docs",
    label: "Docs",
    emoji: "📄",
  }
];

export interface VectorstoreSpec {
  key: string;
  storeType: string;
  name: string;
  emoji?: string;
  settings?: Record<string, any>;
  mode: "create" | "load";
  status?: "none" | "loading" | "ready" | "error";
}

export interface VectorstoreListContainerProps {
  initStoreItems?: VectorstoreSpec[];
  onItemsChange?: (
    newItems: VectorstoreSpec[],
    oldItems: VectorstoreSpec[],
  ) => void;
}

export type VectorstoreListContainerRef = Record<string, never>;

const VectorstoreListItem: React.FC<{
  storeItem: VectorstoreSpec;
  onRemove: (key: string) => void;
  onSettingsUpdate: (key: string, newSettings: any) => void;
  onModeToggle: (key: string) => void;
}> = ({ storeItem, onRemove, onSettingsUpdate, onModeToggle }) => {
  const schemaEntry = VectorstoreSchemas[storeItem.storeType] || {
    schema: {},
    uiSchema: {},
    description: "",
    fullName: "",
  };
  const { schema, uiSchema, fullName, description } = schemaEntry;

  const [settingsModalOpen, { open, close }] = useDisclosure(false);

  const getStatusColor = (status?: string) => {
    switch (status) {
      case "ready":
        return "green";
      case "loading":
        return "yellow";
      case "error":
        return "red";
      default:
        return "gray";
    }
  };

  return (
    <Card shadow="sm" p="sm" withBorder mt="xs">
      <Group position="apart" align="center">
        <div>
          <Group spacing="xs">
            <Text size="sm" weight={600}>
              {storeItem.emoji ? storeItem.emoji + " " : ""}
              {storeItem.name}
            </Text>
            <Switch
              size="xs"
              onLabel="CREATE"
              offLabel="LOAD"
              checked={storeItem.mode === "create"}
              onChange={() => onModeToggle(storeItem.key)}
            />
          </Group>
          <Text size="xs" color="dimmed">
            {fullName || description || ""}
          </Text>
        </div>
        <Group spacing="xs">
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              backgroundColor: getStatusColor(storeItem.status),
            }}
          />
          <ActionIcon variant="subtle" onClick={open} title="Open Settings">
            <IconSettings size={16} />
          </ActionIcon>
          <ActionIcon
            color="red"
            variant="subtle"
            onClick={() => onRemove(storeItem.key)}
            title="Remove this vectorstore"
          >
            <IconTrash size={16} />
          </ActionIcon>
        </Group>
      </Group>

      <Modal
        opened={settingsModalOpen}
        onClose={close}
        title="Vectorstore Settings"
        size="md"
      >
        {schema && Object.keys(schema).length > 0 ? (
          <Form
            schema={schema}
            uiSchema={uiSchema}
            formData={storeItem.settings}
            onChange={(evt) => onSettingsUpdate(storeItem.key, evt.formData)}
            validator={validator as any}
            liveValidate
            noHtml5Validate
          />
        ) : (
          <Text size="sm" color="dimmed">
            (No custom settings for this vectorstore.)
          </Text>
        )}
      </Modal>
    </Card>
  );
};

const VectorstoreListContainer = forwardRef<
  VectorstoreListContainerRef,
  VectorstoreListContainerProps
>((props, ref) => {
  const [storeItems, setStoreItems] = useState<VectorstoreSpec[]>(
    props.initStoreItems || [],
  );
  const oldItemsRef = useRef<VectorstoreSpec[]>(storeItems);

  useImperativeHandle(ref, () => ({}));

  const notifyItemsChanged = useCallback(
    (newItems: VectorstoreSpec[]) => {
      props.onItemsChange?.(newItems, oldItemsRef.current);
      oldItemsRef.current = newItems;
    },
    [props.onItemsChange],
  );

  const handleRemoveStore = useCallback(
    (key: string) => {
      const newItems = storeItems.filter((s) => s.key !== key);
      setStoreItems(newItems);
      notifyItemsChanged(newItems);
    },
    [storeItems, notifyItemsChanged],
  );

  const handleSettingsUpdate = useCallback(
    (key: string, newSettings: any) => {
      const newItems = storeItems.map((s) =>
        s.key === key ? { ...s, settings: newSettings } : s,
      );
      setStoreItems(newItems);
      notifyItemsChanged(newItems);
    },
    [storeItems, notifyItemsChanged],
  );

  const handleModeToggle = useCallback(
    (key: string) => {
      const newItems = storeItems.map((s) =>
        s.key === key
          ? {
              ...s,
              mode:
                s.mode === "create" ? ("load" as const) : ("create" as const),
            }
          : s,
      );
      setStoreItems(newItems);
      notifyItemsChanged(newItems);
    },
    [storeItems, notifyItemsChanged],
  );

  const addStore = useCallback(
    (storeType: string) => {
      const store = vectorstoreTypes.find((t) => t.value === storeType);
      if (!store) return;

      const newItem: VectorstoreSpec = {
        key: uuid(),
        storeType: store.value,
        name: store.label,
        emoji: store.emoji,
        settings: {},
        mode: "create",
        status: "none",
      };
      const newItems = [...storeItems, newItem];
      setStoreItems(newItems);
      notifyItemsChanged(newItems);
    },
    [storeItems, notifyItemsChanged],
  );

  const [menuOpened, setMenuOpened] = useState(false);

  return (
    <div style={{ border: "1px dashed #ccc", borderRadius: 6, padding: 8 }}>
      <Group position="apart" mb="xs">
        <Text weight={500} size="sm">
          Vectorstores
        </Text>

        <Menu
          opened={menuOpened}
          onChange={setMenuOpened}
          position="bottom-end"
          withinPortal
        >
          <Menu.Target>
            <Button
              size="xs"
              variant="light"
              leftIcon={<IconPlus size={14} />}
              onClick={() => setMenuOpened((o) => !o)}
            >
              Add +
            </Button>
          </Menu.Target>
          <Menu.Dropdown>
            {vectorstoreTypes.map((store) => (
              <Menu.Item
                key={store.value}
                icon={store.emoji ? <Text>{store.emoji}</Text> : undefined}
                onClick={() => {
                  addStore(store.value);
                  setMenuOpened(false);
                }}
              >
                {store.label}
              </Menu.Item>
            ))}
          </Menu.Dropdown>
        </Menu>
      </Group>

      {storeItems.length === 0 ? (
        <Text size="xs" color="dimmed">
          No vectorstores added.
        </Text>
      ) : (
        storeItems.map((item) => (
          <VectorstoreListItem
            key={item.key}
            storeItem={item}
            onRemove={handleRemoveStore}
            onSettingsUpdate={handleSettingsUpdate}
            onModeToggle={handleModeToggle}
          />
        ))
      )}
    </div>
  );
});

VectorstoreListContainer.displayName = "VectorstoreListContainer";
export default VectorstoreListContainer;
