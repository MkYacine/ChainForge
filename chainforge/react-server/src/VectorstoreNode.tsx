import React, { useState, useEffect, useCallback, useContext } from "react";
import { Handle, Position } from "reactflow";
import { Status } from "./StatusIndicatorComponent";
import { AlertModalContext } from "./AlertModal";
import BaseNode from "./BaseNode";
import NodeLabel from "./NodeLabelComponent";
import useStore from "./store";
import VectorstoreListContainer, {
  VectorstoreSpec,
} from "./VectorstoreListComponent";

interface VectorstoreNodeProps {
  data: {
    title?: string;
    stores?: VectorstoreSpec[];
    refresh?: boolean;
  };
  id: string;
}

const VectorstoreNode: React.FC<VectorstoreNodeProps> = ({ data, id }) => {
  const nodeDefaultTitle = "Vectorstore Node";
  const nodeIcon = "🗄️";

  const pullInputData = useStore((s) => s.pullInputData);
  const setDataPropsForNode = useStore((s) => s.setDataPropsForNode);
  const pingOutputNodes = useStore((s) => s.pingOutputNodes);

  const showAlert = useContext(AlertModalContext);

  const [storeItems, setStoreItems] = useState<VectorstoreSpec[]>(
    data.stores || [],
  );
  const [status, setStatus] = useState<Status>(Status.NONE);

  // On refresh
  useEffect(() => {
    if (data.refresh) {
      setDataPropsForNode(id, { refresh: false });
      setStatus(Status.NONE);
    }
  }, [data.refresh, id, setDataPropsForNode]);

  // Track changes in vectorstores
  const handleStoreItemsChange = useCallback(
    (newItems: VectorstoreSpec[], _oldItems: VectorstoreSpec[]) => {
      setStoreItems(newItems);
      setDataPropsForNode(id, { stores: newItems });
      if (status === Status.READY) setStatus(Status.WARNING);
    },
    [id, status, setDataPropsForNode],
  );

  // The main vectorstore setup function
  const setupVectorstores = useCallback(async () => {
    if (storeItems.length === 0) {
      showAlert?.("No vectorstores configured!");
      return;
    }

    setStatus(Status.LOADING);

    // Get chunks from upstream node if any stores are in "create" mode
    const hasCreateMode = storeItems.some((store) => store.mode === "create");
    if (hasCreateMode) {
      try {
        const inputData = pullInputData(["chunks"], id);
        if (!inputData.chunks?.length) {
          showAlert?.(
            "No chunks found. Connect a ChunkingNode for 'create' mode.",
          );
          setStatus(Status.ERROR);
          return;
        }
      } catch (error) {
        console.error(error);
        // It's okay if there's no input when all stores are in "load" mode
        if (hasCreateMode) {
          showAlert?.("No input chunks found. Is ChunkingNode connected?");
          setStatus(Status.ERROR);
          return;
        }
      }
    }

    // TODO: Actual vectorstore setup logic will go here
    // For now, just simulate setup
    const updatedItems = storeItems.map((item) => ({
      ...item,
      status: "ready" as const,
    }));

    setStoreItems(updatedItems);
    setDataPropsForNode(id, {
      stores: updatedItems,
      vectorstores: updatedItems.map((store) => ({
        id: store.key,
        type: store.storeType,
        config: store.settings,
      })),
    });

    pingOutputNodes(id);
    setStatus(Status.READY);
  }, [
    id,
    storeItems,
    pullInputData,
    setDataPropsForNode,
    showAlert,
    pingOutputNodes,
  ]);

  return (
    <BaseNode nodeId={id} classNames="vectorstore-node">
      <Handle
        type="target"
        position={Position.Left}
        id="chunks"
        style={{ top: "50%" }}
      />

      <NodeLabel
        title={data.title || nodeDefaultTitle}
        nodeId={id}
        icon={nodeIcon}
        status={status}
        handleRunClick={setupVectorstores}
        runButtonTooltip="Setup/connect vectorstores"
      />

      <VectorstoreListContainer
        initStoreItems={data.stores || []}
        onItemsChange={handleStoreItemsChange}
      />

      <Handle
        type="source"
        position={Position.Right}
        id="vectorstores"
        style={{ top: "50%" }}
      />
    </BaseNode>
  );
};

export default VectorstoreNode;
