use std::collections::BTreeMap;

const MAGIC: &[u8] = b"\x89MCAP0\r\n";
const OP_SCHEMA: u8 = 0x03;
const OP_CHANNEL: u8 = 0x04;
const OP_MESSAGE: u8 = 0x05;
const OP_CHUNK: u8 = 0x06;
const OP_DATA_END: u8 = 0x0F;

#[derive(Debug, Clone)]
pub struct Channel {
    pub topic: String,
    pub schema_name: String,
    pub encoding: String,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub channel: u16,
    pub log_time: u64,
    pub data: Vec<u8>,
}

#[derive(Debug, Default)]
pub struct Recording {
    pub channels: BTreeMap<u16, Channel>,
    pub messages: Vec<Message>,
}

impl Recording {
    pub fn topic_messages<'a>(&'a self, topic: &'a str) -> impl Iterator<Item = &'a Message> + 'a {
        self.messages.iter().filter(move |m| self.channels.get(&m.channel).is_some_and(|c| c.topic == topic))
    }

    pub fn topic_of(&self, message: &Message) -> Option<&str> {
        self.channels.get(&message.channel).map(|c| c.topic.as_str())
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.bytes.len()).ok_or("MCAP record is truncated")?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, String> {
        let n = self.u32()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }

    fn done(&self) -> bool {
        self.pos >= self.bytes.len()
    }
}

#[derive(Default)]
struct Reader {
    schemas: BTreeMap<u16, String>,
    pending: BTreeMap<u16, (u16, String, String)>,
    out: Recording,
}

impl Reader {
    fn records(&mut self, bytes: &[u8], wanted: &dyn Fn(&str) -> bool) -> Result<bool, String> {
        let mut c = Cursor { bytes, pos: 0 };
        while !c.done() {
            let op = c.u8()?;
            let len = c.u64()? as usize;
            let body = c.take(len)?;
            match op {
                OP_SCHEMA => self.schema(body)?,
                OP_CHANNEL => self.channel(body)?,
                OP_MESSAGE => self.message(body, wanted)?,
                OP_CHUNK => self.chunk(body, wanted)?,
                OP_DATA_END => return Ok(true),
                _ => {}
            }
        }
        Ok(false)
    }

    fn schema(&mut self, body: &[u8]) -> Result<(), String> {
        let mut c = Cursor { bytes: body, pos: 0 };
        let id = c.u16()?;
        let name = c.string()?;
        self.schemas.insert(id, name);
        self.resolve();
        Ok(())
    }

    fn channel(&mut self, body: &[u8]) -> Result<(), String> {
        let mut c = Cursor { bytes: body, pos: 0 };
        let id = c.u16()?;
        let schema_id = c.u16()?;
        let topic = c.string()?;
        let encoding = c.string()?;
        self.pending.insert(id, (schema_id, topic, encoding));
        self.resolve();
        Ok(())
    }

    fn resolve(&mut self) {
        let ids: Vec<u16> = self.pending.keys().copied().collect();
        for id in ids {
            let schema_id = self.pending[&id].0;
            if schema_id == 0 || self.schemas.contains_key(&schema_id) {
                let (_, topic, encoding) = self.pending.remove(&id).unwrap();
                let schema_name = self.schemas.get(&schema_id).cloned().unwrap_or_default();
                self.out.channels.insert(id, Channel { topic, schema_name, encoding });
            }
        }
    }

    fn message(&mut self, body: &[u8], wanted: &dyn Fn(&str) -> bool) -> Result<(), String> {
        let mut c = Cursor { bytes: body, pos: 0 };
        let channel = c.u16()?;
        let _sequence = c.u32()?;
        let log_time = c.u64()?;
        let _publish_time = c.u64()?;
        let keep = self.out.channels.get(&channel).is_some_and(|ch| wanted(&ch.topic));
        if keep {
            self.out.messages.push(Message { channel, log_time, data: body[c.pos..].to_vec() });
        }
        Ok(())
    }

    fn chunk(&mut self, body: &[u8], wanted: &dyn Fn(&str) -> bool) -> Result<(), String> {
        let mut c = Cursor { bytes: body, pos: 0 };
        let _start = c.u64()?;
        let _end = c.u64()?;
        let size = c.u64()? as usize;
        let _crc = c.u32()?;
        let compression = c.string()?;
        let n = c.u64()? as usize;
        let data = c.take(n)?;
        let raw = match compression.as_str() {
            "" => data.to_vec(),
            "zstd" => {
                let mut out = Vec::with_capacity(size);
                let mut decoder = ruzstd::decoding::StreamingDecoder::new(data).map_err(|e| e.to_string())?;
                std::io::Read::read_to_end(&mut decoder, &mut out).map_err(|e| e.to_string())?;
                out
            }
            "lz4" => {
                let mut out = Vec::with_capacity(size);
                let mut decoder = lz4_flex::frame::FrameDecoder::new(data);
                std::io::Read::read_to_end(&mut decoder, &mut out).map_err(|e| e.to_string())?;
                out
            }
            other => return Err(format!("unsupported MCAP chunk compression {other:?}")),
        };
        self.records(&raw, wanted)?;
        Ok(())
    }
}

pub fn read_mcap(bytes: &[u8], wanted: &dyn Fn(&str) -> bool) -> Result<Recording, String> {
    if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
        return Err("not an MCAP file".into());
    }
    let mut reader = Reader::default();
    reader.records(&bytes[MAGIC.len()..], wanted)?;
    reader.out.messages.sort_by_key(|m| m.log_time);
    Ok(reader.out)
}
